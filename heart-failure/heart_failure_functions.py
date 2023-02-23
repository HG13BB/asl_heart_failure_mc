#helper functions for heart failure project

import os
import io 
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def upload_blob_from_string(bucket_name, contents, destination_blob_name):
    """Uploads a file to the bucket."""

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The contents to upload to the file
    # contents = "these are my contents"

    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(contents)

    print(
        f"{destination_blob_name} with contents {contents} uploaded to {bucket_name}."
    )


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

def download_blob_into_string(bucket_name, blob_name):
    """Downloads a blob into string."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # blob_name = "storage-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(blob_name)
    contents = blob.download_as_string()

    print(
        "Downloaded storage object {} from bucket {} as the following string: {}.".format(
            blob_name, bucket_name, contents
        )
    )    


def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))    

    
def split_and_scale(heart_failure_df, random_seed=42, train_split=.8, val_split=.1, test_split=.1, num_rows_to_duplicate=100):
    
    train, test = train_test_split(heart_failure_df, random_state=random_seed, test_size=(test_split - test_split*val_split), stratify="HeartDisease")
    train, val = train_test_split(train, random_state=random_seed*2, test_size=val_split, stratify="HeartDisease")

    print(len(train), "train examples")
    print(len(val), "validation examples")
    print(len(test), "test examples")
    
    #expected values "TRAIN", "VALIDATE", "TEST", "UNASSIGNED"
    train['split'] = "TRAIN"
    val['split'] = "VALIDATE"
    test['split'] = "TEST"

    print("train:\n")
    #display(HTML(train.to_html(max_rows=10)))
    train.head()

    print("val:")
    #display(HTML(val.to_html(max_rows=10)))
    val.head()

    print("test:")
    #display(HTML(test.to_html(max_rows=10)))
    test.head()
    
    
    train_scaled = pd.concat([train, train.sample(n=math.ceil(num_rows_to_duplicate*train_split), random_state=random_seed)],axis=0)
    val_scaled = pd.concat([val, val.sample(n=math.ceil(num_rows_to_duplicate*val_split), random_state=random_seed)],axis=0)
    test_scaled = pd.concat([test, test.sample(n=math.ceil(num_rows_to_duplicate*test_split), random_state=random_seed)],axis=0)
    scaled_dataset = pd.concat([train_scaled,val_scaled, test_scaled], axis=0)
    
    return train_scaled, val_scaled, test_scaled, scaled_dataset

def engineer_features(input_file: string):
    numeric_cols = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak"
    ]

    #bucketized_cols = [""]

    # indicator columns,Categorical features
    categorical_cols = [
        "Sex",
        "ChestPainType",
        "FastingBS", 
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope"
    ]
    
    scaled_feature_engineered_dataset = pd.DataFrame()

    for column in categorical_cols:
        onehot = pd.get_dummies(scaled_dataset[column], prefix=column, prefix_sep=".")
        scaled_feature_engineered_dataset = pd.concat([scaled_feature_engineered_dataset, onehot], axis=1)

    for column in numeric_cols:
        scaled_feature_engineered_dataset[column] = (scaled_dataset[column] - scaled_dataset[column].min()) / (scaled_dataset[column].max() - scaled_dataset[column].min()) 

    bins = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scaled_feature_engineered_dataset['binned_Oldpeak'] = np.searchsorted(bins, scaled_dataset['Oldpeak'].values)    

    scaled_feature_engineered_dataset["split"]=scaled_dataset["split"]
    scaled_feature_engineered_dataset["HeartDisease"]=scaled_dataset["HeartDisease"]

    return scaled_feature_engineered_dataset
    
def process_data(input_file_uri: string, scaled_output_prefix="scaled": string, feature_engineered_output_prefix="scaled-engineered") string:
    bucket = input_file_uri.split("/")[2]
    input_object_name = "/".join(input_file_uri.split("/")[3:])
    
    
    input_string = download_blob_into_string(bucket, input_object_name)
    heart_failure_df = pd.read_csv(io.StringIO(input_string), on_bad_lines="skip")
    heart_failure_df.head()
    heart_failure_df.describe()
    
    train_scaled, val_scaled, test_scaled, scaled_dataset=split_and_scale(heart_failure_df)
    
    print(len(train_scaled), "scaled train examples")
    print(len(val_scaled), "validation examples")
    print(len(test_scaled), "test examples")
    print(len(scaled_dataset), "total examples")
    print("scaled_dataset:")
    scaled_dataset.head()
    scaled_dataset.describe()
    
    upload_blob_from_string(bucket, train_scaled.to_csv(encoding="utf-8", index=False), ..
    upload_blob_from_string(bucket, val_scaled.to_csv(encoding="utf-8", index=False)
    upload_blob_from_string(bucket, test_scaled.to_csv(encoding="utf-8", index=False)
    upload_blob_from_string(bucket, scaled_dataset.to_csv(encoding="utf-8", index=False)
    
    scaled_feature_engineered_dataset=engineer_features(scaled_dataset)
    print("scaled_feature_engineered_dataset:", len(scaled_feature_engineered_dataset), "total examples")
    #display(HTML(scaled_feature_engineered_dataset.to_html(max_rows=10)))
    scaled_feature_engineered_dataset.head()                        
    scaled_feature_engineered_dataset.describe()
    
    upload_blob_from_string(bucket, scaled_feature_engineered_dataset.to_csv(encoding="utf-8", index=False)
    #output_object_name = "/".join(output_file_uri.split("/")[3:])
    output_file_uri = "blah"
    
    return output_file_uri
    