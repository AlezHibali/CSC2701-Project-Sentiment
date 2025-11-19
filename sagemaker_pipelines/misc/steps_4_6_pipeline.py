import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile

def get_deploy_only_pipeline():
    boto_sess = boto3.Session()
    region = boto_sess.region_name
    pipeline_sess = PipelineSession()

    role = "arn:aws:iam::210446372308:role/service-role/AmazonSageMaker-ExecutionRole-20251031T195225"

    base_job_prefix = "twitter-roberta-sentiment"

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    input_model_artifact = ParameterString(
        name="InputModelArtifactUri",
        default_value="s3://sagemaker-us-east-1-210446372308/pipelines-xsasddiq8dxp-FineTuneRoBERTaModel-S4U3opVGCz/output/model.tar.gz",
    )
    evaluation_output_uri = ParameterString(
        name="EvaluationOutputUri",
        default_value="s3://sagemaker-us-east-1-210446372308/twitter-roberta-sentiment-Pipeline/xsasddiq8dxp/PreprocessSentimentData/output/test_data/",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.80)

    model = HuggingFaceModel(
        model_data=input_model_artifact,
        role=role,
        image_uri=(
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
            "huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04"
        ),
        sagemaker_session=pipeline_sess,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(on="/", values=[evaluation_output_uri, "evaluation.json"]),
            content_type="application/json"
        )
    )

    model_package_group_name = f"{base_job_prefix}-Registry"
    register_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    step_register_model = ModelStep(
        name="RegisterNewModelVersion",
        step_args=register_args,
    )

    deploy_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type.default_value,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/deploy",
        role=role,
        sagemaker_session=pipeline_sess,
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
        ],
    )

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=0.85,
        right=accuracy_threshold
    )

    step_conditional = ConditionStep(
        name="CheckModelAccuracy",
        conditions=[cond_gte],
        if_steps=[step_register_model, step_deploy],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=f"{base_job_prefix}-DeployOnlyPipeline",
        parameters=[
            processing_instance_type,
            input_model_artifact,
            evaluation_output_uri,
            model_approval_status,
            accuracy_threshold,
        ],
        steps=[step_conditional],
        sagemaker_session=pipeline_sess,
    )


    return pipeline

if __name__ == "__main__":
    pipeline = get_deploy_only_pipeline()

    print("Upserting deploy-only pipeline definition...")
    pipeline.upsert(
        role_arn="arn:aws:iam::210446372308:role/service-role/AmazonSageMaker-ExecutionRole-20251031T195225"
    )
    print("✅ Pipeline definition successfully upserted.")

    print("\nStarting pipeline execution...")
    execution = pipeline.start()
    print(f"🚀 Pipeline execution started with ARN: {execution.arn}")

    print("\nMonitor progress in SageMaker Studio.")
    print(execution.describe())