# Twitter Sentiment Analysis -- Frontend + SageMaker Pipeline

This repository contains a complete **serverless sentiment analysis
application** built with:

-   **Next.js (App Router)** -- Frontend UI & API routes
-   **AWS SageMaker Serverless Endpoint** -- Hosts a fine-tuned
    **Twitter Roberta** sentiment model
-   **AWS App Runner** -- Deployment for the Next.js frontend
-   **SageMaker Pipelines** -- Training, evaluation, and deployment (under `sagemaker_pipelines/`)

------------------------------------------------------------------------

TODO: features
TODO: add screenshots here

## 📂 Repository Structure

    /frontend             # Next.js frontend + API to SageMaker
    /sagemaker_pipelines  # Code for training + deploying Roberta to SageMaker

------------------------------------------------------------------------

## Frontend (Next.js)

Located in `frontend/`.\
The frontend provides:

-   A simple UI to input a tweet
-   Calls a **Next.js API route** at `/api/classify`
-   The API route forwards the request to the SageMaker endpoint

### Running locally

``` bash
cd frontend
npm install
npm run dev
```

This opens up an interface at:
    http://localhost:3000

------------------------------------------------------------------------

## SageMaker (model evaluation + serverless inference endpoint)

Located in `sagemaker_pipelines/`.

This folder contains:

- A **SageMaker Pipeline** for evaluating the fine-tuned RoBERTa sentiment model  
- Full pipeline for training, evaluating, and deploying Roberta
- Scripts for deploy only, evaluate only, and data-handling only pipelines

### Install Requirement
```
cd sagemaker_pipelines
pip install -r requirements.yxy
```

### Full Pipeline
`python run_pipeline.py`

### Partial Pipeline
`python misc/deploy_only_pipeline.py`

`python misc/evaluate_only_pipeline.py`

`python misc/steps_4_6_pipeline.py`

------------------------------------------------------------------------

## ✔️ Summary

TODO: a table to present all steps of MLOps
