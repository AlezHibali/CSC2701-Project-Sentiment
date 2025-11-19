import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

role = "arn:aws:iam::210446372308:role/SageMaker-Execution-Role"
sagemaker_session = sagemaker.Session()

model_data = "s3://sagemaker-us-east-1-210446372308/pipelines-6roqsy1sjtko-FineTuneRoBERTaModel-lFRwrx0bD3/output/model.tar.gz"

image_uri = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.6.0-transformers4.49.0-cpu-py312-ubuntu22.04"
)

model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    image_uri=image_uri,
    sagemaker_session=sagemaker_session,
)

serverless_config = ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=2)

predictor = model.deploy(
    endpoint_name="manual-test-serverless",
    serverless_inference_config=serverless_config,
)