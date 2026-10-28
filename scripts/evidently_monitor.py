import argparse
import base64
import glob
import json
import os
from datetime import datetime, timedelta, timezone

import boto3
import mlflow
import pandas as pd
from evidently import BinaryClassification, Dataset, DataDefinition, Report
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mlflow-tracking-uri", required=True)
    p.add_argument("--mlflow-experiment", required=True)
    p.add_argument("--drift-threshold", type=float, default=0.05)
    p.add_argument("--lookback-days", type=int, default=7,
                   help="Only consider capture records within the last N days. 0 disables the filter.")
    p.add_argument("--critical-features", default="",
                   help="Comma-separated features that alert individually if drifted (e.g. 'campaign,age_18-29').")
    p.add_argument("--min-f1", type=float, default=0.0,
                   help="If >0, alert when ClassificationPreset F1 falls below this value.")
    p.add_argument("--sns-topic-arn", default="")
    p.add_argument("--endpoint-name", default="")
    return p.parse_args()


def load_capture(capture_dir, feature_names, lookback_days):
    cutoff = None
    if lookback_days and lookback_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows, skipped = [], 0
    for path in glob.glob(os.path.join(capture_dir, "**", "*.jsonl"), recursive=True):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                meta = rec.get("eventMetadata", {})
                if cutoff is not None:
                    ts_str = meta.get("inferenceTime", "")
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except Exception:
                        ts = None
                    if ts is not None and ts < cutoff:
                        skipped += 1
                        continue
                cap = rec.get("captureData", {})
                inp, out = cap.get("endpointInput", {}), cap.get("endpointOutput", {})
                raw_in = inp.get("data", "")
                if inp.get("encoding") == "BASE64":
                    raw_in = base64.b64decode(raw_in).decode("utf-8")
                raw_out = out.get("data", "")
                if out.get("encoding") == "BASE64":
                    raw_out = base64.b64decode(raw_out).decode("utf-8")
                rows.append({
                    "inference_id": meta.get("inferenceId", ""),
                    "input": raw_in.strip(),
                    "proba": float(raw_out.strip()) if raw_out.strip() else None,
                })
    if skipped:
        print(f"Skipped {skipped} capture records older than {lookback_days} day(s).")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    features = pd.DataFrame(df["input"].str.split(",").tolist(), columns=feature_names).astype(float)
    df["prediction"] = (df["proba"] > 0.5).astype(int)
    return pd.concat([df[["inference_id", "proba", "prediction"]], features], axis=1)


def drifted_share(report_dict):
    for m in report_dict.get("metrics", []):
        if m.get("metric_name", "").startswith("DriftedColumnsCount"):
            v = m.get("value")
            return float(v.get("share", 0.0)) if isinstance(v, dict) else 0.0
    return 0.0


def drifted_critical(report_dict, critical):
    """Return list of critical features whose ValueDrift exceeded their threshold."""
    hits = []
    if not critical:
        return hits
    for m in report_dict.get("metrics", []):
        if not m.get("metric_name", "").startswith("ValueDrift"):
            continue
        cfg = m.get("config", {}) or {}
        column, threshold, val = cfg.get("column"), cfg.get("threshold"), m.get("value")
        if column in critical and threshold is not None and val is not None:
            try:
                if float(val) > float(threshold):
                    hits.append(column)
            except (TypeError, ValueError):
                pass
    return hits


def all_drifted(report_dict):
    """Return list of (column, value) for every column whose ValueDrift exceeded its threshold, sorted desc."""
    hits = []
    for m in report_dict.get("metrics", []):
        if not m.get("metric_name", "").startswith("ValueDrift"):
            continue
        cfg = m.get("config", {}) or {}
        column, threshold, val = cfg.get("column"), cfg.get("threshold"), m.get("value")
        if column and threshold is not None and val is not None:
            try:
                fv = float(val)
                if fv > float(threshold):
                    hits.append((column, fv))
            except (TypeError, ValueError):
                pass
    return sorted(hits, key=lambda x: x[1], reverse=True)


def classification_f1(report_dict):
    for m in report_dict.get("metrics", []):
        if m.get("metric_name", "").startswith("F1Score") and not m.get("metric_name", "").startswith("F1ByLabel"):
            try:
                return float(m.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def publish_alert(topic_arn, subject, message):
    if not topic_arn:
        return
    try:
        # Extract region from the SNS topic ARN (arn:aws:sns:<region>:<acct>:<name>)
        region = topic_arn.split(":")[3] if topic_arn.count(":") >= 3 else os.environ.get("AWS_REGION", "us-east-1")
        boto3.client("sns", region_name=region).publish(TopicArn=topic_arn, Subject=subject, Message=message)
        print(f"SNS alert published: {subject}")
    except Exception as e:
        print(f"[WARN] SNS publish skipped: {e}")


def main():
    args = parse_args()
    base_in, base_out = "/opt/ml/processing/input", "/opt/ml/processing/output"
    os.makedirs(base_out, exist_ok=True)

    reference = pd.read_csv(f"{base_in}/baseline/baseline.csv")
    captured = load_capture(f"{base_in}/capture", feature_names=reference.columns.tolist(),
                            lookback_days=args.lookback_days)
    if captured is None or len(captured) == 0:
        print(f"[WARN] No capture records found within the last {args.lookback_days} day(s). "
              f"Nothing to compare — exiting successfully.")
        return

    production = captured[reference.columns.tolist()].copy()
    critical = [c.strip() for c in args.critical_features.split(",") if c.strip()]

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"scheduled_monitoring_{ts}"):
        mlflow.log_params({
            "reference_size": len(reference),
            "current_size": len(production),
            "drift_threshold": args.drift_threshold,
            "lookback_days": args.lookback_days,
            "critical_features": args.critical_features,
            "min_f1": args.min_f1,
            "endpoint_name": args.endpoint_name,
        })

        snap = Report(metrics=[DataDriftPreset(), DataSummaryPreset()]).run(
            reference_data=Dataset.from_pandas(reference, data_definition=DataDefinition()),
            current_data=Dataset.from_pandas(production, data_definition=DataDefinition()),
        )
        html_path, json_path = f"{base_out}/data_drift_{ts}.html", f"{base_out}/data_drift_{ts}.json"
        snap.save_html(html_path)
        snap.save_json(json_path)
        mlflow.log_artifact(html_path, "evidently_report_data_drift_html")
        mlflow.log_artifact(json_path, "evidently_report_data_drift_json")

        drift_dict = snap.dict()
        share = drifted_share(drift_dict)
        mlflow.log_metric("DriftedColumnsCount.share", share)
        print(f"Drifted columns share: {share:.3f} (threshold={args.drift_threshold})")

        all_hits = all_drifted(drift_dict)
        if all_hits:
            print(f"All drifted features: {all_hits}")

        critical_hits = drifted_critical(drift_dict, critical)
        if critical_hits:
            print(f"Critical features drifted: {critical_hits}")

        f1 = None
        gt_path = f"{base_in}/ground_truth/ground_truth.csv"
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path)
            eval_df = captured.merge(gt, on="inference_id", how="inner")
            if len(eval_df):
                dd = DataDefinition(classification=[
                    BinaryClassification(target="target", prediction_labels="prediction", pos_label=1)
                ])
                cls_snap = Report(metrics=[ClassificationPreset()]).run(
                    reference_data=None,
                    current_data=Dataset.from_pandas(eval_df, data_definition=dd),
                )
                cls_html = f"{base_out}/classification_{ts}.html"
                cls_snap.save_html(cls_html)
                mlflow.log_artifact(cls_html, "evidently_classification_report_html")
                f1 = classification_f1(cls_snap.dict())
                if f1 is not None:
                    mlflow.log_metric("F1Score", f1)
                    print(f"Classification F1: {f1:.3f} (min_f1={args.min_f1})")

        # Consolidate triggers
        reasons = []
        if share > args.drift_threshold:
            reasons.append(f"drifted columns share {share:.2%} > threshold {args.drift_threshold:.2%}")
        if critical_hits:
            reasons.append(f"critical features drifted: {', '.join(critical_hits)}")
        if args.min_f1 > 0 and f1 is not None and f1 < args.min_f1:
            reasons.append(f"F1 {f1:.3f} < min_f1 {args.min_f1:.3f}")

        if reasons:
            run = mlflow.active_run()
            run_id = run.info.run_id if run else "n/a"
            experiment_id = run.info.experiment_id if run else "n/a"
            mlflow_ui_hint = (
                f"Open SageMaker Studio -> MLflow -> Experiment '{args.mlflow_experiment}' "
                f"-> run_id {run_id}"
            )
            drifted_block = ""
            if all_hits:
                drifted_block = "Drifted features (feature: score):\n  - " + "\n  - ".join(
                    f"{col}: {score:.4f}" for col, score in all_hits
                ) + "\n\n"

            publish_alert(
                args.sns_topic_arn,
                subject=f"[SageMaker] Model monitoring alert — {args.endpoint_name}",
                message=(
                    f"Endpoint: {args.endpoint_name}\n"
                    f"Run timestamp: {ts}\n\n"
                    f"Triggers:\n  - " + "\n  - ".join(reasons) + "\n\n"
                    + drifted_block +
                    f"MLflow experiment: {args.mlflow_experiment} (id {experiment_id})\n"
                    f"MLflow run_id:     {run_id}\n"
                    f"{mlflow_ui_hint}\n"
                ),
            )


if __name__ == "__main__":
    main()
