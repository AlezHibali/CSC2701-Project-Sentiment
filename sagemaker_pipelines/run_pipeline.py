import os
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace, HuggingFaceModel, HuggingFaceProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


def _resolve_role():
    # Prefer env var if set, then SDK, else fallback to a literal
    env_role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if env_role:
        return env_role
    try:
        return sagemaker.get_execution_role()
    except Exception:
        return "arn:aws:iam::210446372308:role/service-role/AmazonSageMaker-ExecutionRole-20251031T195225"


def _latest_approved_model_and_artifact(sm_client, group_name: str):
    """
    Returns (latest_model_package_arn, model_data_url) for the newest Approved package.
    """
    resp = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )
    if not resp["ModelPackageSummaryList"]:
        raise ValueError(f"No Approved model packages found in group {group_name}")
    latest_pkg_arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]

    desc = sm_client.describe_model_package(ModelPackageName=latest_pkg_arn)
    model_data_url = (
        desc.get("InferenceSpecification", {})
            .get("Containers", [{}])[0]
            .get("ModelDataUrl")
        or desc.get("ModelDataUrl")
    )
    if not model_data_url:
        raise ValueError(f"Could not resolve ModelDataUrl for model package {latest_pkg_arn}")
    return latest_pkg_arn, model_data_url


def get_sagemaker_pipeline():
    """
    Defines and returns the complete SageMaker MLOps pipeline.
    """
    # --- 1. Sessions / base config ---
    pipeline_session = PipelineSession()
    boto_session = boto3.Session()
    region = boto_session.region_name
    role = _resolve_role()
    base_job_prefix = "twitter-roberta-sentiment"

    sm = boto3.client("sagemaker", region_name=region)

    # --- 2. Pipeline Parameters ---
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.p3.2xlarge"
    )

    input_data_s3_uri = ParameterString(
        name="InputDataS3Uri",
        default_value=(
            "s3://amazon-sagemaker-210446372308-us-east-1-32b98cd66f6c/"
            "dzd-44j4g2plc46dlz/673glz4luibvdz/shared/data/train/unprocessed/train_000.jsonl"
        ),
    )
    validation_data_s3_uri = ParameterString(
        name="ValidationDataS3Uri",
        default_value=(
            "s3://amazon-sagemaker-210446372308-us-east-1-32b98cd66f6c/"
            "dzd-44j4g2plc46dlz/673glz4luibvdz/shared/data/validation"
        ),
    )
    test_data_s3_uri = ParameterString(
        name="TestDataS3Uri",
        default_value=(
            "s3://amazon-sagemaker-210446372308-us-east-1-32b98cd66f6c/"
            "dzd-44j4g2plc46dlz/673glz4luibvdz/shared/data/test"
        ),
    )
    processed_prefix = ParameterString(
        name="ProcessedPrefix",
        default_value=(
            "s3://amazon-sagemaker-210446372308-us-east-1-32b98cd66f6c/"
            "dzd-44j4g2plc46dlz/673glz4luibvdz/shared/data/train/processed"
        ),
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.10)

    # --- 3. Resolve latest Approved Registry model NOW (no Lambda) ---
    model_package_group_name = f"{base_job_prefix}-Registry"
    latest_pkg_arn, latest_model_data_url = _latest_approved_model_and_artifact(sm, model_package_group_name)

    # --- 4. Steps ---

    # STEP 1: Preprocess train only
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/preprocess",
        role=role,
        sagemaker_session=pipeline_session,
    )
    step_process = ProcessingStep(
        name="PreprocessSentimentData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data_s3_uri, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ],
        code="scripts/preprocess.py",
    )

    # STEP 2: Fine-tune starting from latest Registry model
    hyperparameters = {
        "epochs": 1,
        "train_batch_size": 32,
        # Fallback only; registry model will override via 'pretrained' channel
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    }
    hf_estimator = HuggingFace(
        entry_point="train.py",
        source_dir="scripts",
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        transformers_version="4.28",
        pytorch_version="2.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        sagemaker_session=pipeline_session,
    )
    step_train = TrainingStep(
        name="FineTuneRoBERTaModel",
        estimator=hf_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri
            ),
            "validation": TrainingInput(s3_data=validation_data_s3_uri),
            "pretrained": TrainingInput(s3_data=latest_model_data_url),
        },
    )

    # STEP 3: Evaluation on dedicated test set (12.3k)
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    evaluation_processor = HuggingFaceProcessor(
        transformers_version="4.28",
        pytorch_version="2.0",
        py_version="py310",
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}/evaluate",
        role=role,
        sagemaker_session=pipeline_session,
    )
    step_args = evaluation_processor.run(
        code="evaluate.py",
        source_dir="scripts",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(source=test_data_s3_uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        arguments=[
            "--model-path", "/opt/ml/processing/model",
            "--test-data-path", "/opt/ml/processing/test",
            "--output-path", "/opt/ml/processing/evaluation",
        ],
    )
    step_evaluate = ProcessingStep(
        name="EvaluateModelPerformance",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # STEP 4: Register model (uses evaluation metrics)
    model = HuggingFaceModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        image_uri=(
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
            "huggingface-pytorch-inference:2.6.0-transformers4.49.0-cpu-py312-ubuntu22.04"
        ),
        sagemaker_session=pipeline_session,
    )
    evaluation_output = step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(on="/", values=[evaluation_output, "evaluation.json"]),
            content_type="application/json"
        )
    )
    register_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=[processing_instance_type.default_value],
        transform_instances=[processing_instance_type.default_value],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register_model = ModelStep(
        name="RegisterNewModelVersion",
        step_args=register_args,
    )

    # STEP 5: Deploy to serverless + move consumed batch
    deploy_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/deploy",
        role=role,
        sagemaker_session=pipeline_session,
    )
    step_deploy = ProcessingStep(
        name="DeployModelToServerlessEndpoint",
        processor=deploy_processor,
        code="scripts/deploy.py",
        job_arguments=[
            "--model-package-arn", step_register_model.properties.ModelPackageArn,
            "--endpoint-name", "arn:aws:sagemaker:us-east-1:210446372308:endpoint/twitter-roberta-sentiment-serverless",
            "--region", region,
            "--role", role,
            "--source-s3-uri", input_data_s3_uri,
            "--processed-prefix", processed_prefix,
        ]
    )

    # STEP 6: Conditional deploy when accuracy >= threshold
    accuracy_value = JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path="multiclass_classification_metrics.accuracy.value"
    )
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=accuracy_value,
        right=accuracy_threshold
    )
    step_conditional = ConditionStep(
        name="CheckModelAccuracy",
        conditions=[cond_gte],
        if_steps=[step_register_model, step_deploy],
        else_steps=[],
    )

    # --- Assemble pipeline ---
    pipeline = Pipeline(
        name=f"{base_job_prefix}-Pipeline",
        parameters=[
            processing_instance_type,
            training_instance_type,
            input_data_s3_uri,
            validation_data_s3_uri,
            test_data_s3_uri,
            model_approval_status,
            accuracy_threshold,
            processed_prefix,
        ],
        steps=[step_process, step_train, step_evaluate, step_conditional],
        sagemaker_session=pipeline_session,
    )
    return pipeline


if __name__ == "__main__":
    pipeline = get_sagemaker_pipeline()

    print("Upserting pipeline definition...")
    role_to_use = _resolve_role()
    # upsert requires a literal role_arn when running locally
    pipeline.upsert(role_arn=role_to_use)
    print("Pipeline definition successfully upserted.")

    print("\nStarting pipeline execution...")
    execution = pipeline.start()
    print(f"Pipeline execution started with ARN: {execution.arn}")

    print("\nYou can monitor the execution in the SageMaker Studio UI.")
    execution.describe()