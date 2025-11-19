import boto3
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile

ROLE_ARN = "arn:aws:iam::210446372308:role/service-role/AmazonSageMaker-ExecutionRole-20251031T195225"

INSTANCE_TYPE = "ml.g4dn.xlarge"
INSTANCE_COUNT = 1

BASE_JOB_PREFIX = "twitter-roberta-sentiment"
PIPELINE_NAME = f"{BASE_JOB_PREFIX}-EvalOnlyPipeline"

MODEL_S3_URI = (
    "s3://sagemaker-us-east-1-210446372308/"
    "pipelines-xsasddiq8dxp-FineTuneRoBERTaModel-S4U3opVGCz/output/model.tar.gz"
)
TEST_DATA_S3_URI = (
    "s3://sagemaker-us-east-1-210446372308/"
    "twitter-roberta-sentiment-Pipeline/xsasddiq8dxp/PreprocessSentimentData/output/test_data/"
)

def main():
    pipeline_sess = PipelineSession()

    evaluation_processor = HuggingFaceProcessor(
        transformers_version="4.28",
        pytorch_version="2.0",
        py_version="py310",
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        base_job_name=f"{BASE_JOB_PREFIX}/evaluate",
        role=ROLE_ARN,
        sagemaker_session=pipeline_sess,
    )

    step_args = evaluation_processor.run(
        code="evaluate.py",
        source_dir="scripts",
        inputs=[
            ProcessingInput(source=MODEL_S3_URI, destination="/opt/ml/processing/model"),
            ProcessingInput(source=TEST_DATA_S3_URI, destination="/opt/ml/processing/test"),
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
        property_files=[
            PropertyFile(
                name="EvaluationReport",
                output_name="evaluation",
                path="evaluation.json",
            )
        ],
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[],
        steps=[step_evaluate],
        sagemaker_session=pipeline_sess,
    )

    print("Upserting eval-only pipeline definition...")
    pipeline.upsert(role_arn=ROLE_ARN)
    print("✅ Pipeline definition upserted.")

    print("\nStarting eval-only pipeline execution...")
    execution = pipeline.start()
    print(f"🚀 Execution ARN: {execution.arn}")
    print("\nDescribe execution:")
    print(execution.describe())

if __name__ == "__main__":
    main()
