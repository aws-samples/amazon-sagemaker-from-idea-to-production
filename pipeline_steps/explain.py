"""Explainability step - SHAP feature attribution and standardized bias metrics.

One implementation, two call sites:

  * notebook 01 runs it as a **managed job** by applying the ``@remote``
    decorator to this function: ``remote(**cfg)(explain)(...)``
  * notebook 03 wraps the *same* function as a **pipeline step**:
    ``step(explain, name=...)(...)``

Notebook 01 applies ``@remote`` to this function; notebook 03 applies ``@step``.
The function body is shared, so both paths produce the same SHAP and fairness
report.

Why SHAP and pandas: both are open-source building blocks with no service
dependency - SHAP for feature attribution and the standardized fairness-metric
formulas (class imbalance, difference in positive proportions, disparate impact)
computed directly with pandas - so the report is reproducible anywhere the model
itself runs.

Why ``TreeExplainer``: the model under analysis is a plain XGBoost booster, so
tree SHAP applies and is **exact** - no sampling, and fast enough to explain the
whole test set (~4k rows x ~60 one-hot features in well under a second). Kernel
SHAP is only needed when the thing being explained is an opaque pipeline.

Extra dependencies over the BYOC image (which already ships xgboost, pandas,
numpy, mlflow and s3fs): ``shap`` and ``matplotlib`` - see
``explainability/requirements.txt``.
"""
import json
import os
import tempfile
from time import gmtime, strftime

import mlflow
import numpy as np
import pandas as pd

try:
    from pipeline_steps.runtime_utils import log_runtime_info
except ImportError:  # running as a flat module inside a job
    def log_runtime_info():
        pass


# --------------------------------------------------------------------------- #
# Standardized bias metrics (pandas implementation of the usual fairness metrics)
# --------------------------------------------------------------------------- #
def compute_bias_metrics(labels, predicted_labels, in_facet, positive_label=1):
    """Compute a core set of pre-training and post-training bias metrics.

    ``in_facet`` is a boolean mask selecting the facet of interest (group ``d``);
    the complement is the comparison group (group ``a``).

    Pre-training (from true labels):
      * CI   - Class Imbalance
      * DPL  - Difference in Positive Proportions in true Labels
    Post-training (from predicted labels):
      * DPPL - Difference in Positive Proportions in Predicted Labels
      * DI   - Disparate Impact

    These are the standard definitions used in the fairness literature; see e.g.
    Feldman et al. (2015) for Disparate Impact and the fairlearn documentation
    (https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html)
    for the difference-in-positive-proportions family.
    """
    labels = np.asarray(labels).ravel()
    predicted_labels = np.asarray(predicted_labels).ravel()
    is_d = np.asarray(in_facet).ravel().astype(bool)
    is_a = ~is_d

    n_a, n_d = int(is_a.sum()), int(is_d.sum())

    def _pos_rate(mask, values):
        n = int(mask.sum())
        return float((values[mask] == positive_label).sum()) / n if n else float("nan")

    q_a, q_d = _pos_rate(is_a, labels), _pos_rate(is_d, labels)
    p_a, p_d = _pos_rate(is_a, predicted_labels), _pos_rate(is_d, predicted_labels)

    return {
        "n_comparison_group": n_a,
        "n_facet_group": n_d,
        "true_positive_rate_comparison": q_a,
        "true_positive_rate_facet": q_d,
        "pred_positive_rate_comparison": p_a,
        "pred_positive_rate_facet": p_d,
        "CI_class_imbalance": (n_a - n_d) / (n_a + n_d) if (n_a + n_d) else float("nan"),
        "DPL_diff_positive_labels": q_a - q_d,
        "DPPL_diff_positive_predicted_labels": p_a - p_d,
        "DI_disparate_impact": (p_d / p_a) if p_a else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _save_global_plots(shap_values, features_df, out_dir, max_display):
    """Write the beeswarm summary plot and the mean-|SHAP| bar plot."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no display inside a job
    import matplotlib.pyplot as plt
    import shap

    paths = []
    for plot_type, filename, title in [
        (None, "shap_summary_beeswarm.png", "SHAP summary (impact on model output, log-odds)"),
        ("bar", "shap_feature_importance_bar.png", "Global feature importance (mean |SHAP|)"),
    ]:
        path = os.path.join(out_dir, filename)
        shap.summary_plot(
            shap_values, features_df, plot_type=plot_type, max_display=max_display, show=False
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def _save_waterfall(shap_row, expected_value, data_row, feature_names, out_path, max_display):
    """Write a single-instance waterfall plot explaining one prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explanation = shap.Explanation(
        values=np.asarray(shap_row),
        base_values=float(expected_value),
        data=np.asarray(data_row),
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def _upload_report(paths, s3_prefix):
    """Copy the report files to S3 so they are reachable outside MLflow too."""
    import boto3

    bucket, _, key_prefix = s3_prefix.replace("s3://", "").partition("/")
    s3 = boto3.client("s3")
    for path in paths:
        s3.upload_file(path, bucket, f"{key_prefix}/{os.path.basename(path)}")


# --------------------------------------------------------------------------- #
# The step
# --------------------------------------------------------------------------- #
def explain(
    mlflow_run_id,
    test_x_data_s3_path,
    test_y_data_s3_path,
    feature_names,
    output_s3_prefix,
    facet_columns=None,
    max_display=20,
    pipeline_run_id=None,
):
    """Explain the model logged by ``mlflow_run_id`` and log the report to MLflow.

    Args:
        mlflow_run_id: run whose logged model is analysed. The model is resolved
            through the run's ``outputs.model_outputs``, the same way
            ``pipeline_steps/register.py`` resolves it.
        test_x_data_s3_path: header-less CSV of features, no label column.
        test_y_data_s3_path: header-less CSV of ground-truth labels.
        feature_names: column names for the test features, in order. The
            preprocessing writes header-less CSVs, so the names travel as data.
        output_s3_prefix: S3 prefix the report files are copied to.
        facet_columns: feature columns defining the facet for the bias metrics.
            A row is in the facet when any of them is non-zero, so a set of
            one-hot columns such as the ``age_60-69`` / ``age_70-plus`` bins can
            express "customer aged 60+". Bias metrics are skipped when omitted.
        max_display: features shown per plot.
        pipeline_run_id: parent MLflow run to nest under, when called as a step.

    Returns:
        dict of report locations, headline metrics and run ids.
    """
    try:
        suffix = strftime("%d-%H-%M-%S", gmtime())

        pipeline_run = mlflow.start_run(run_id=pipeline_run_id) if pipeline_run_id else None
        run = mlflow.start_run(run_name=f"explainability-{suffix}", nested=bool(pipeline_run_id))
        log_runtime_info()
        mlflow.set_tag("analysis", "shap-explainability-and-bias")

        # ---------------------------------------------------------------- model
        client = mlflow.MlflowClient()
        source_run = client.get_run(mlflow_run_id)
        model_outputs = source_run.outputs.model_outputs if source_run.outputs else []
        if not model_outputs:
            raise RuntimeError(
                f"MLflow run {mlflow_run_id} has no logged model outputs - nothing to explain."
            )
        model_id = model_outputs[0].model_id

        local_model_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_id}")
        booster = mlflow.xgboost.load_model(local_model_dir)
        print(f"## Explaining logged model {model_id} from run {mlflow_run_id}")

        # ----------------------------------------------------------------- data
        test_x = pd.read_csv(test_x_data_s3_path, header=None)
        test_y = pd.read_csv(test_y_data_s3_path, header=None).iloc[:, 0].to_numpy()

        if len(feature_names) != test_x.shape[1]:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but the test features have "
                f"{test_x.shape[1]} columns."
            )
        test_x.columns = list(feature_names)
        print(f"## Explaining {test_x.shape[0]} rows x {test_x.shape[1]} features")

        # ----------------------------------------------------------------- SHAP
        import shap
        import xgboost as xgb

        explainer = shap.TreeExplainer(booster)
        shap_values = np.asarray(explainer.shap_values(test_x))
        expected_value = float(np.ravel(explainer.expected_value)[0])
        probabilities = booster.predict(xgb.DMatrix(test_x))
        predicted_labels = (probabilities > 0.5).astype(int)

        print(f"## E[y] (log-odds of the average prediction): {expected_value:.4f}")

        out_dir = tempfile.mkdtemp()
        artifacts = _save_global_plots(shap_values, test_x, out_dir, max_display)

        # Explain the single most confident positive prediction.
        top_index = int(np.argmax(probabilities))
        artifacts.append(
            _save_waterfall(
                shap_values[top_index],
                expected_value,
                test_x.iloc[top_index].to_numpy(),
                test_x.columns,
                os.path.join(out_dir, "shap_waterfall_top_prediction.png"),
                max_display,
            )
        )

        # Per-instance table: prediction + SHAP values + features.
        shap_table = pd.concat(
            [
                pd.Series(predicted_labels, name="prediction"),
                pd.Series(probabilities, name="probability_score"),
                pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in test_x.columns]),
                test_x.reset_index(drop=True),
            ],
            axis=1,
        )
        shap_csv = os.path.join(out_dir, "shap_values.csv")
        shap_table.to_csv(shap_csv, index=False)
        artifacts.append(shap_csv)

        mean_abs_shap = (
            pd.DataFrame(shap_values, columns=test_x.columns)
            .abs()
            .mean()
            .sort_values(ascending=False)
        )
        print("## Top features by mean |SHAP|:")
        for feature, value in mean_abs_shap.head(max_display).items():
            print(f"##   {feature:<30}{value:.5f}")

        # ----------------------------------------------------------------- bias
        bias_metrics = {}
        if facet_columns:
            missing = [c for c in facet_columns if c not in test_x.columns]
            if missing:
                raise ValueError(f"facet_columns not present in the features: {missing}")
            in_facet = test_x[list(facet_columns)].to_numpy().sum(axis=1) > 0
            bias_metrics = compute_bias_metrics(test_y, predicted_labels, in_facet)
            bias_json = os.path.join(out_dir, "bias_metrics.json")
            with open(bias_json, "w") as f:
                json.dump(bias_metrics, f, indent=2)
            artifacts.append(bias_json)
            print(f"## Bias metrics for facet {list(facet_columns)}: {bias_metrics}")

        # --------------------------------------------------------------- MLflow
        mlflow.log_params(
            {
                "explained_model_id": model_id,
                "explained_run_id": mlflow_run_id,
                "n_explained_instances": int(test_x.shape[0]),
                "n_features": int(test_x.shape[1]),
                "explainer": "shap.TreeExplainer",
                "facet_columns": ",".join(facet_columns) if facet_columns else "",
            }
        )
        mlflow.log_metric("shap_expected_value_logodds", expected_value)
        for feature, value in mean_abs_shap.items():
            mlflow.log_metric(f"mean_abs_shap_{feature}", float(value))
        for name, value in bias_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                mlflow.log_metric(f"bias_{name}", float(value))
        for path in artifacts:
            mlflow.log_artifact(path, artifact_path="explainability_report")

        # ------------------------------------------------------------ S3 report
        report_s3_prefix = f"{output_s3_prefix}/explainability"
        _upload_report(artifacts, report_s3_prefix)
        print(f"## Report copied to {report_s3_prefix}")

        print("## Explainability complete. Exiting.")
        return {
            "report_s3_prefix": report_s3_prefix,
            "shap_values_s3_path": f"{report_s3_prefix}/shap_values.csv",
            "explained_model_id": model_id,
            "shap_expected_value_logodds": expected_value,
            "top_features": mean_abs_shap.head(max_display).index.tolist(),
            "bias_metrics": bias_metrics,
            "explainability_run_id": run.info.run_id,
            "pipeline_run_id": pipeline_run.info.run_id if pipeline_run else "",
        }

    except Exception as e:
        print(f"Exception in explain step: {e}")
        raise e
    finally:
        while mlflow.active_run():
            mlflow.end_run()
