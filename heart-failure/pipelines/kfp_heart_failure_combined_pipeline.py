# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Kubeflow Combined AutoMl and Simple DNN Heart Failure Pipeline."""

import os

import sys
from typing import NamedTuple

from google.cloud import aiplatform as vertex
from google_cloud_pipeline_components import \
    aiplatform as vertex_pipeline_components
from kfp.v2 import compiler
from kfp.v2.dsl import Artifact, Input, Metrics, Output, component

from google_cloud_pipeline_components.experimental.custom_job import (
    CustomTrainingJobOp,
)

from google_cloud_pipeline_components.aiplatform import (
    AutoMLTabularTrainingJobRunOp,
    EndpointCreateOp,
    ModelDeployOp,
    TabularDatasetCreateOp,
    ModelUploadOp,
)
from kfp.v2 import dsl

PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT = os.getenv("PROJECT")
REGION = os.getenv("REGION")

DATASET_URI = os.getenv("DATASET_URI")
OUTPUT_URI = os.getenv("OUTPUT_URI")
PIPELINE_NAME = os.getenv("PIPELINE_NAME", "kfp-heartfailure-combined-dnn")
DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", PIPELINE_NAME)
SERVING_MACHINE_TYPE = os.getenv("SERVING_MACHINE_TYPE", "n1-standard-4")
TRAINING_CONTAINER_IMAGE_URI = os.getenv("TRAINING_CONTAINER_IMAGE_URI")
SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "HeartDisease")
SPLIT_COLUMN = os.getenv("SPLIT_COLUMN", "split")
OPTIMIZATION_OBJECTIVE = os.getenv("OPTIMIZATION_OBJECTIVE", "maximize-au-roc")
#BUDGET_MILLI_NODE_HOURS = os.getenv("BUDGET_MILLI_NODE_HOURS", "2000")
BUDGET_MILLI_NODE_HOURS = os.getenv("BUDGET_MILLI_NODE_HOURS", "1000")

@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
)
def interpret_automl_evaluation_metrics(
    region: str, model: Input[Artifact], metrics: Output[Metrics]
):
    """'
    For a list of available regression metrics, go here: gs://google-cloud-aiplatform/schema/modelevaluation/regression_metrics_1.0.0.yaml.

    More information on available metrics for different types of models: https://cloud.google.com/vertex-ai/docs/predictions/online-predictions-automl
    """

    import google.cloud.aiplatform.gapic as gapic

    # Get a reference to the Model Service client
    client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}

    model_service_client = gapic.ModelServiceClient(client_options=client_options)

    model_resource_name = model.metadata["resourceName"]

    model_evaluations = model_service_client.list_model_evaluations(
        parent=model_resource_name
    )
    model_evaluation = list(model_evaluations)[0]

    available_metrics = [
        "meanAbsoluteError",
        "meanAbsolutePercentageError",
        "rSquared",
        "rootMeanSquaredError",
        "rootMeanSquaredLogError",
    ]
    output = dict()
    for x in available_metrics:
        val = model_evaluation.metrics.get(x)
        output[x] = val
        #metrics.log_metric(str(x), float(val)) #get a conversation error from None
        metrics.log_metric(str(x), val)

    metrics.log_metric("framework", "AutoML")
    print(output)


@dsl.pipeline(
    name=f"{PIPELINE_NAME}-vertex-pipeline",
    description=f"Vertex Pipeline for {PIPELINE_NAME}",
    pipeline_root=PIPELINE_ROOT,
)
def create_pipeline():

    dataset_create_task = TabularDatasetCreateOp(
        display_name=DISPLAY_NAME,
        gcs_source=DATASET_URI,
        project=PROJECT,
    )

    automl_training_task = AutoMLTabularTrainingJobRunOp(
        project=PROJECT,
        display_name=DISPLAY_NAME,
        optimization_prediction_type="classification",
        dataset=dataset_create_task.outputs["dataset"],
        target_column=TARGET_COLUMN,
        predefined_split_column_name=SPLIT_COLUMN,
        optimization_objective=OPTIMIZATION_OBJECTIVE,
        budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,
    )
    
    automl_model = automl_training_task.outputs["model"]

    # Analyzes evaluation AutoML metrics using a custom component.
    automl_eval_op = interpret_automl_evaluation_metrics(
        region=REGION, model=automl_model
    )
    automl_eval_metrics = automl_eval_op.outputs["metrics"]
    print("automl_eval_metrics:", automl_eval_metrics)
    
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINING_CONTAINER_IMAGE_URI,
                "args": [
                    f"--dataset_uri={DATASET_URI}",
                    f"--output_uri={OUTPUT_URI}",
                    "--epochs=100",
                    "--batch_size=100",
                    "--lr=.001",
                ],
            },
        }
    ]
    
    training_task = CustomTrainingJobOp(
        project=PROJECT,
        location=REGION,
        display_name=f"{PIPELINE_NAME}-training-job",
        worker_pool_specs=worker_pool_specs,
        base_output_directory=BASE_OUTPUT_DIR,
    )
    
    model_upload_task = ModelUploadOp(
        project=PROJECT,
        display_name=f"{PIPELINE_NAME}-upload-job",
        #artifact_uri=OUTPUT_URI,
        artifact_uri=f"{BASE_OUTPUT_DIR}/model",
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
    ) 
    model_upload_task.after(training_task)

    endpoint_create_task = EndpointCreateOp(
        project=PROJECT,
        display_name=f"{PIPELINE_NAME}-endpoint-job",
        description="Best Heart Failure model",
    )
    endpoint_create_task.after(model_upload_task)

    model_deploy_task = ModelDeployOp(  # pylint: disable=unused-variable
        model=model_upload_task.outputs["model"],
        endpoint=endpoint_create_task.outputs["endpoint"],
        deployed_model_display_name=DISPLAY_NAME,
        dedicated_resources_machine_type=SERVING_MACHINE_TYPE,
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        traffic_split={'0': 100}
        #enable_access_logging=True, #comment out because of failure
    )
