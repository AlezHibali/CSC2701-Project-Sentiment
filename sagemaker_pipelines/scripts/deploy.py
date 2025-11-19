import argparse
import logging
import boto3
import time
from urllib.parse import urlparse
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def update_serverless_endpoint(
    model_package_arn,
    endpoint_name,
    region,
    role,
    memory_size,
    max_concurrency,
    source_s3_uri=None,
    processed_prefix=None
):
    try:
        sm_client = boto3.client("sagemaker", region_name=region)
        s3 = boto3.client("s3", region_name=region)
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

        model_name = f"{endpoint_name}-model-{timestamp}"
        logging.info(f"Creating SageMaker Model: {model_name}")
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={"ModelPackageName": model_package_arn},
            ExecutionRoleArn=role
        )

        endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
        logging.info(f"Creating Endpoint Configuration: {endpoint_config_name}")

        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                    "ServerlessConfig": {
                        "MemorySizeInMB": memory_size,
                        "MaxConcurrency": max_concurrency
                    },
                    "InitialVariantWeight": 1.0
                }
            ]
        )

        logging.info(f"Updating endpoint '{endpoint_name}' to use config '{endpoint_config_name}'")
        try:
            sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logging.info(f"✅ Updated existing endpoint: {endpoint_name}")
        except ClientError as e:
            msg = str(e)
            if "Could not find endpoint" in msg or "Could not find endpoint" in getattr(e, "response", {}).get("Error", {}).get("Message", ""):
                sm_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )
                logging.info(f"✅ Created new endpoint: {endpoint_name}")
            else:
                raise

        waiter = sm_client.get_waiter("endpoint_in_service")
        logging.info("Waiting for endpoint to be InService...")
        waiter.wait(EndpointName=endpoint_name)
        logging.info(f"✅ Endpoint '{endpoint_name}' is now InService.")

        # Move the consumed training batch after successful deployment
        if source_s3_uri and processed_prefix:
            logging.info(f"Moving training batch from {source_s3_uri} to {processed_prefix}")
            parsed_src = urlparse(source_s3_uri)
            if parsed_src.scheme != "s3":
                raise ValueError("--source-s3-uri must be an s3:// URI")
            src_bucket = parsed_src.netloc
            src_key = parsed_src.path.lstrip("/")
            filename = src_key.split("/")[-1]

            parsed_dst = urlparse(processed_prefix)
            if parsed_dst.scheme != "s3":
                raise ValueError("--processed-prefix must be an s3:// URI")
            dst_bucket = parsed_dst.netloc
            dst_prefix = parsed_dst.path.lstrip("/").rstrip("/")
            dst_key = f"{dst_prefix}/{filename}"

            s3.copy_object(Bucket=dst_bucket, CopySource={"Bucket": src_bucket, "Key": src_key}, Key=dst_key)
            s3.delete_object(Bucket=src_bucket, Key=src_key)
            logging.info(f"✅ Moved s3://{src_bucket}/{src_key} -> s3://{dst_bucket}/{dst_key}")

    except Exception as e:
        logging.error(f"❌ Failed to update endpoint: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-arn", type=str, required=True)
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--memory-size", type=int, default=2048)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--source-s3-uri", type=str, required=False, help="The exact unprocessed training batch file used")
    parser.add_argument("--processed-prefix", type=str, required=False, help="s3://.../.../processed")
    args = parser.parse_args()

    endpoint_name_from_arn = args.endpoint_name.split("/")[-1]

    update_serverless_endpoint(
        args.model_package_arn,
        endpoint_name_from_arn,
        args.region,
        args.role,
        args.memory_size,
        args.max_concurrency,
        args.source_s3_uri,
        args.processed_prefix
    )