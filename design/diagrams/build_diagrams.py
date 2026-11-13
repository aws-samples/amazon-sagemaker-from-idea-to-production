"""Generate the workshop diagrams with AWS Architecture Icons.

Run from the repository root:

    python design/diagrams/build_diagrams.py

The diagram of notebook 05 (img/deployment-project-architecture-aws.svg) is the style
reference and isn't generated here.
"""
from pathlib import Path

from svgkit import (APP_INTEGRATION, CONTAINERS, DEVELOPER_TOOLS, GENERAL, SAGEMAKER, STORAGE,
                    Diagram)

ROOT = Path(__file__).resolve().parents[2]


def workshop_flow():
    d = Diagram(1800, 760, "Workshop flow",
                "Run 00-start-here first. The main path goes through steps 01 to 05. A fast track goes from "
                "00 directly to step 03 and continues with steps 04 and 05. Optional notebooks in the "
                "additional-topics folder extend steps 04 and 05. Always finish with 99-clean-up.")
    d.heading("Workshop flow")
    # legend
    d.add('<rect x="1060" y="34" width="34" height="24" rx="5" class="node-required"/>')
    d.text(1104, 53, "Required", "legend")
    d.add('<rect x="1210" y="34" width="34" height="24" rx="5" class="node-optional"/>')
    d.text(1254, 53, "Optional notebook", "legend")
    d.arrow("M 1440 46 L 1490 46", "arrow-highlight")
    d.text(1500, 53, "Fast track", "legend")

    steps = [
        ("00", ["Set up the", "environment"], ["MLflow app, container", "image, dataset"], "Arch_Amazon-Elastic-Container-Registry_64", GENERAL, True),
        ("01", ["Experiment in", "a notebook"], ["pandas, XGBoost,", "MLflow tracking"], "Res_Amazon-SageMaker_Notebook_48", SAGEMAKER, False),
        ("02", ["Scale with", "SageMaker AI jobs"], ["@remote,", "ModelTrainer"], "Res_Amazon-SageMaker_Train_48", SAGEMAKER, False),
        ("03", ["ML pipeline and", "model registry"], ["@step,", "SageMaker Pipelines"], "Arch_Amazon-SageMaker_64", SAGEMAKER, False),
        ("04", ["Deploy and", "monitor"], ["endpoint,", "Evidently, MLflow"], "Res_Amazon-SageMaker_Model_48", SAGEMAKER, False),
        ("05", ["Automate the", "deployment"], ["SageMaker project,", "CI/CD pipeline"], "Arch_AWS-CodePipeline_64", DEVELOPER_TOOLS, False),
        ("99", ["Clean up", "resources"], ["avoid costs in", "your account"], "Res_Gear_48_Light", GENERAL, True),
    ]
    y, h, w = 170, 250, 222
    for i, (number, label, detail, icon, accent, required) in enumerate(steps):
        x = 40 + i * 250
        d.node(x, y, w, h, label, detail, icon, accent, eyebrow=number,
               cls="node-required" if required else "node")
        if i:
            d.arrow(f"M {x - 28} {y + h / 2} L {x - 2} {y + h / 2}")
    # Orthogonal fast-track route with rounded corners: up from 00, across above the steps, down into 03
    d.arrow("M 151 168 L 151 132 Q 151 118 165 118 L 887 118 Q 901 118 901 132 L 901 164", "arrow-highlight",
            "Fast track: pipelines only, 00 → 03 → 04 → 05", 526, 106, label_cls="highlight-label")

    optional = [
        (1033, 236, "A/B testing", ["A-B-testing: model", "variants on one endpoint"], 1151),
        (1283, 236, "Build and deploy", ["05, 06: CI/CD project", "with GitHub repositories"], 1401),
    ]
    for x, w2, label, detail, source_x in optional:
        d.node(x, 520, w2, 140, label, detail, None, None, cls="node-optional")
        d.arrow(f"M {source_x} {y + h} L {x + w2 / 2} 516", "arrow-dashed")
    d.text(1033, 700, "Optional notebooks are in the additional-topics folder. Run them after the step they extend.", "node-detail")
    d.save(ROOT / "design/workshop-flow.svg")


def training_options():
    d = Diagram(1800, 720, "SageMaker AI training options",
                "Four ways to train a model with SageMaker AI: built-in algorithms, your script on a SageMaker "
                "framework container, your own container, and AWS Marketplace algorithms. In all options SageMaker "
                "AI provides the managed infrastructure. This workshop uses its own container.")
    d.heading("SageMaker AI training options")
    d.add('<rect x="1130" y="34" width="34" height="24" rx="5" class="node-dark"/>')
    d.text(1174, 53, "You provide", "legend")
    d.add('<rect x="1310" y="34" width="34" height="24" rx="5" class="provided"/>')
    d.text(1354, 53, "SageMaker AI provides", "legend")

    columns = [
        ("Built-in algorithms", "Arch_Amazon-SageMaker_64",
         [("Data and hyperparameters", True), ("Algorithm and container", False), ("Managed infrastructure", False)],
         ["XGBoost, Linear Learner, k-means, and", "more, without training code"]),
        ("Script mode", "Res_Amazon-SageMaker_Train_48",
         [("Data and training script", True), ("Framework container", False), ("Managed infrastructure", False)],
         ["PyTorch, TensorFlow, XGBoost,", "scikit-learn, Hugging Face containers", "run your script"]),
        ("Your own container", "Arch_Amazon-Elastic-Container-Registry_64",
         [("Data and training script", True), ("Container image in Amazon ECR", True), ("Managed infrastructure", False)],
         ["Any framework and dependencies. The", "image follows the SageMaker AI container", "contract: /opt/ml layout, train and serve"]),
        ("AWS Marketplace", "Arch_AWS-Marketplace_Light_64",
         [("Data and hyperparameters", True), ("Third-party algorithm", False), ("Managed infrastructure", False)],
         ["Algorithms and models that you", "subscribe to"]),
    ]
    for i, (title, icon, layers, detail) in enumerate(columns):
        x = 40 + i * 435
        this_workshop = i == 2
        d.group(x, 100, 410, 500, title, icon, "subgroup" if this_workshop else "group")
        if this_workshop:
            d.badge(x + 222, 84, "This workshop")
        for j, (layer, yours) in enumerate(layers):
            ly = 190 + j * 86
            d.add(f'<rect x="{x + 24}" y="{ly}" width="362" height="64" rx="10" class="{"node-dark" if yours else "provided"}"/>')
            d.text(x + 205, ly + 39, layer, "node-label-light" if yours else "node-label", "middle")
        for j, line in enumerate(detail):
            d.text(x + 205, 480 + 22 * j, line, "node-detail", "middle")
    d.text(40, 660, "In all options SageMaker AI starts the instances, copies the data from S3, runs the training, "
                    "saves the model to S3, and stops the instances.", "legend")
    d.save(ROOT / "img/sagemaker-training-options.svg")


def one_function_four_ways():
    d = Diagram(1800, 720, "One function, four ways to run it",
                "The train function in pipeline_steps/train_fn.py runs in four ways: called in the notebook, "
                "wrapped with remote as a SageMaker training job, as the entry script of a ModelTrainer training "
                "job in your container, and wrapped with step as a pipeline step. A native TrainingStep runs the "
                "same ModelTrainer in a pipeline.")
    d.heading("One function, four ways to run it")
    d.node(650, 90, 500, 200, "pipeline_steps/train_fn.py", ["the same code everywhere"], "Res_Source-Code_48_Light",
           GENERAL, code=["train()"])
    ways = [
        ("NOTEBOOK 02", "Call it in the notebook", "train(...)", ["quick check of the code", "before you start a job"], "Res_Amazon-SageMaker_Notebook_48"),
        ("NOTEBOOK 02", "Run it as a job", "remote(...)(train)", ["SageMaker AI runs the function", "and returns its result"], "Res_Amazon-SageMaker_Train_48"),
        ("NOTEBOOK 02", "Run the file in your container", "ModelTrainer(...).train()", ["script mode: main() reads input", "channels and hyperparameters"], "Arch_Amazon-Elastic-Container-Registry_64"),
        ("NOTEBOOK 03", "Run it in a pipeline", "step(...)(train)", ["pipeline step, wired to the other", "steps by return values"], "Arch_Amazon-SageMaker_64"),
    ]
    for i, (eyebrow, label, code, detail, icon) in enumerate(ways):
        x = 40 + i * 440
        d.node(x, 360, 400, 270, label, detail, icon, CONTAINERS if i == 2 else SAGEMAKER, code=[code], eyebrow=eyebrow)
    for start_x, end_x in [(720, 240), (840, 680), (960, 1120), (1080, 1560)]:
        d.arrow(f"M {start_x} 290 C {start_x} 330, {end_x} 320, {end_x} 356")
    d.node(1040, 660, 720, 50, "", [], None, None, cls="node-optional")
    d.text(1400, 692, "Notebook 03: a native TrainingStep runs the same ModelTrainer", "node-detail", "middle")
    d.arrow("M 1120 630 L 1120 656", "arrow-dashed")
    d.text(40, 692, "preprocess(), explain(), evaluate(), and register() run the same way in the notebook, with remote(), and with step().", "node-detail")
    d.save(ROOT / "img/one-function-four-ways.svg")


def register_deployable_model():
    d = Diagram(1600, 640, "Register a deployable model",
                "Log inference.py and an inference specification on the MLflow logged model, then register it in "
                "MLflow. The automatic model registration creates a SageMaker model package version with the "
                "inference specification. An endpoint created from the model package downloads the model files "
                "directly from the MLflow artifact store.")
    d.heading("Register a deployable model")
    d.node(40, 100, 380, 330, "MLflow logged model", ["S3 artifact store of the MLflow app", "+ inference specification"],
           "Arch_Amazon-Simple-Storage-Service_64", STORAGE,
           code=["MLmodel, model.ubj, conda.yaml", "code/inference.py"])
    # actions 1 and 2
    d.step(470, 190, 1)
    d.text(500, 186, "log_model_artifacts()", "code")
    d.text(500, 210, "add code/inference.py", "node-detail")
    d.step(470, 290, 2)
    d.text(500, 286, "log_inference_specification()", "code")
    d.text(500, 310, "image, model data, environment", "node-detail")
    d.arrow("M 420 240 L 450 240", "line-dashed")
    d.node(860, 100, 300, 260, ["MLflow Model", "Registry"], ["new model version"], "Arch_Amazon-SageMaker_64", SAGEMAKER,
           code=["mlflow.register_model()"], step=3)
    d.arrow("M 800 230 L 856 230")
    d.node(1260, 100, 300, 260, ["SageMaker Model", "Registry"], ["deployable model package", "with the inference specification"],
           "Res_Amazon-SageMaker_Model_48", SAGEMAKER)
    d.arrow("M 1160 230 L 1256 230", label="auto-sync", lx=1208, ly=218)
    d.node(1260, 420, 300, 180, "Endpoint", ["steps 4 and 5"], "Arch_Amazon-SageMaker_64", SAGEMAKER)
    d.arrow("M 1410 360 L 1410 416", label="deploy", lx=1424, ly=394, anchor="start")
    d.arrow("M 1256 510 L 230 510 L 230 434", "arrow-dashed",
            "downloads the model files to /opt/ml/model, no repacking or copies", 740, 498)
    d.save(ROOT / "img/register-deployable-model.svg")


def inference_handlers():
    d = Diagram(1800, 760, "How the endpoint container calls inference.py",
                "The serving app in the container answers GET /ping and POST /invocations on port 8080. At startup "
                "it calls model_fn once with /opt/ml/model to load the model. For each request it calls input_fn to "
                "parse the request body, predict_fn to run the model, and output_fn to build the response body.")
    d.heading("How the endpoint container calls inference.py")
    d.node(40, 300, 220, 220, "Client", ["CSV or JSON request"], "Res_Client_48_Light", GENERAL, code=["invoke_endpoint"])
    d.group(300, 90, 1460, 640, "SageMaker AI endpoint: your container, docker run <image> serve", "Arch_Amazon-SageMaker_64")
    d.node(330, 170, 360, 530, "Serving app", [], "Res_Server_48_Light", CONTAINERS,
           code=["byoc/serving_app.py"])
    lines = [("GET /ping", "code", 360), ("200 when the model is loaded", "node-detail", 386),
             ("POST /invocations", "code", 440), ("calls the request handlers", "node-detail", 466),
             ("Loads the script", "node-detail", 540), ("SAGEMAKER_PROGRAM", "code", 566),
             ("from the folder", "node-detail", 600), ("SAGEMAKER_SUBMIT_DIRECTORY", "code", 626)]
    for text, cls, y in lines:
        d.text(510, y, text, cls, "middle")
    d.arrow("M 260 390 L 326 390")
    d.arrow("M 330 440 L 264 440", label="response", lx=295, ly=468)

    d.group(730, 170, 1000, 530, "Your code: /opt/ml/model/code/inference.py", "Res_Source-Code_48_Light", "subgroup")
    d.text(760, 270, "Once, at startup", "eyebrow")
    d.node(760, 285, 330, 110, "", ["load the model from /opt/ml/model"], None, SAGEMAKER, code=["model_fn(model_dir)"])
    d.node(1150, 285, 260, 110, "", ["kept in memory"], None, SAGEMAKER, code=["model"])
    d.arrow("M 1090 340 L 1146 340")
    d.arrow("M 690 230 C 725 230, 725 340, 756 340")
    d.text(760, 455, "For each request", "eyebrow")
    handlers = [("input_fn(body, type)", ["request body to a numpy array"]),
                ("predict_fn(data, model)", ["run the model: probabilities"]),
                ("output_fn(pred, accept)", ["predictions to the response body"])]
    for i, (code, detail) in enumerate(handlers):
        x = 760 + i * 325
        d.node(x, 470, 300, 110, "", detail, None, SAGEMAKER, code=[code])
        if i:
            d.arrow(f"M {x - 25} 525 L {x - 4} 525")
    d.arrow("M 690 440 C 725 440, 725 525, 756 525")
    d.arrow("M 1280 395 L 1240 466", "arrow-dashed")
    d.arrow("M 1560 580 L 1560 640 L 694 640", label="response body and content type", lx=1130, ly=628)
    d.save(ROOT / "img/inference-handlers.svg")


def deployment_and_monitoring():
    d = Diagram(1800, 1020, "Deployment and monitoring workflow of step 4",
                "The ML pipeline of step 3 registers the model package and writes the baseline to S3. The notebook "
                "approves the model package and deploys it to a real-time endpoint with data capture, sends test "
                "traffic with simulated drift, and stores the ground truth labels in S3. A monitoring pipeline, "
                "started from the notebook or by an optional EventBridge Scheduler schedule, runs a processing job "
                "with Evidently on the baseline, the captured data, and the ground truth. It logs reports and "
                "metrics in the MLflow app and publishes an SNS email alert when a check fails.")
    d.heading("Step 4: deploy and monitor")
    d.arrow("M 1370 46 L 1420 46")
    d.text(1430, 53, "Control", "legend")
    d.arrow("M 1540 46 L 1590 46", "arrow-dashed")
    d.text(1600, 53, "Data", "legend")

    # Part 1
    d.group(20, 90, 1760, 440, "Part 1: Deploy the approved model")
    d.node(44, 160, 250, 340, ["ML pipeline", "from step 3"], ["preprocess, train,", "evaluate, register"], "Arch_Amazon-SageMaker_64", SAGEMAKER)
    d.node(360, 160, 320, 160, "SageMaker Model Registry", ["approve the model package"], "Res_Amazon-SageMaker_Model_48", SAGEMAKER, step=1)
    d.node(360, 340, 320, 160, "Amazon S3", ["baseline and test data"], "Arch_Amazon-Simple-Storage-Service_64", STORAGE)
    d.node(780, 160, 320, 160, "Real-time endpoint", ["data capture of all requests"], "Arch_Amazon-SageMaker_64", SAGEMAKER, step=2)
    d.node(780, 340, 320, 160, "Notebook", ["test traffic with simulated drift"], "Res_Amazon-SageMaker_Notebook_48", SAGEMAKER)
    d.node(1200, 160, 320, 160, "Amazon S3", ["captured data, JSON Lines"], "Arch_Amazon-Simple-Storage-Service_64", STORAGE)
    d.node(1200, 340, 320, 160, "Amazon S3", ["ground truth labels"], "Arch_Amazon-Simple-Storage-Service_64", STORAGE)
    d.arrow("M 294 240 L 356 240")
    d.arrow("M 294 420 L 356 420", "arrow-dashed")
    d.arrow("M 680 240 L 776 240", label="deploy", lx=728, ly=228)
    d.arrow("M 940 340 L 940 324", label="invoke", lx=956, ly=336, anchor="start")
    d.arrow("M 1100 240 L 1196 240", "arrow-dashed")
    d.arrow("M 1100 420 L 1196 420", "arrow-dashed")

    # Part 2
    d.group(20, 570, 1760, 430, "Part 2: Monitor the model")
    d.node(44, 640, 250, 160, "EventBridge Scheduler", ["daily, off by default"], "Res_Amazon-EventBridge_Scheduler_48",
           APP_INTEGRATION, cls="node-optional")
    d.node(44, 820, 250, 160, "Notebook", ["runs the checks first"], "Res_Amazon-SageMaker_Notebook_48", SAGEMAKER,
           code=["pipeline.start()"])
    d.group(360, 640, 580, 340, "Monitoring pipeline", "Arch_Amazon-SageMaker_64", "subgroup")
    d.step(384, 640, 3)
    d.node(390, 720, 520, 230, "Processing job", ["Evidently data drift, data summary, and", "classification quality reports",
                                                  "scikit-learn image + scripts/requirements.txt"],
           None, SAGEMAKER, code=["scripts/evidently_monitor.py"])
    d.arrow("M 294 720 L 356 760", "arrow-dashed")
    d.arrow("M 294 900 L 356 880", label="start", lx=325, ly=820)
    # inputs
    d.arrow("M 520 500 L 520 636", "arrow-dashed", "baseline", 534, 556, "start")
    d.arrow("M 1360 320 L 1360 335 L 1550 335 L 1550 550 L 760 550 L 760 636", "arrow-dashed")
    d.arrow("M 1360 500 L 1360 550", "line-dashed")
    d.text(776, 600, "captured data, ground truth", "flow-label")
    # outputs
    d.node(1010, 640, 300, 160, "SageMaker AI MLflow app", ["reports and metrics per run"], "Arch_Amazon-SageMaker_64", SAGEMAKER, step=4)
    d.node(1380, 640, 300, 160, "MLflow UI", ["compare the monitoring runs"], "Res_Users_48_Light", GENERAL)
    d.node(1010, 820, 300, 160, "Amazon SNS topic", ["alert if a check fails"], "Arch_Amazon-Simple-Notification-Service_64", APP_INTEGRATION, step=5)
    d.node(1380, 820, 300, 160, "Email", ["drift share, critical feature, F1"], "Res_Amazon-Simple-Notification-Service_Email-Notification_48", APP_INTEGRATION)
    d.arrow("M 940 720 L 1006 720", "arrow-dashed")
    d.arrow("M 940 900 L 1006 900")
    d.arrow("M 1310 720 L 1376 720")
    d.arrow("M 1310 900 L 1376 900")
    d.save(ROOT / "img/deployment-and-monitoring.svg")


if __name__ == "__main__":
    workshop_flow()
    training_options()
    one_function_four_ways()
    register_deployable_model()
    inference_handlers()
    deployment_and_monitoring()
    print("Diagrams written")
