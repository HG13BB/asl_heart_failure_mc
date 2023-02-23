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

"""Kubeflow Simple DNN Heart Failure Pipeline."""

import os

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
PIPELINE_NAME = os.getenv("PIPELINE_NAME", "kfp-heartfailure-simple-dnn")
DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", PIPELINE_NAME)
SERVING_MACHINE_TYPE = os.getenv("SERVING_MACHINE_TYPE", "n1-standard-4")
TRAINING_CONTAINER_IMAGE_URI = os.getenv("TRAINING_CONTAINER_IMAGE_URI")
SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR")


@dsl.pipeline(
    name=f"{PIPELINE_NAME}-vertex-pipeline",
    description=f"Vertex Pipeline for {PIPELINE_NAME}",
    pipeline_root=PIPELINE_ROOT,
)
def create_pipeline():

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
        artifact_uri=f"{BASE_OUTPUT_DIR}/model",
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
    ) 
    model_upload_task.after(training_task)

    endpoint_create_task = EndpointCreateOp(
        project=PROJECT,
        display_name=f"{PIPELINE_NAME}-endpoint-job",
        description="Heart Failure Simple DNN model",
    )
    endpoint_create_task.after(model_upload_task)

    model_deploy_task = ModelDeployOp(  # pylint: disable=unused-variable
        model=model_upload_task.outputs["model"],
        endpoint=endpoint_create_task.outputs["endpoint"],
        deployed_model_display_name=DISPLAY_NAME,
        dedicated_resources_machine_type=SERVING_MACHINE_TYPE,
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        #enable_access_logging=True, #comment out because of failure
    )
