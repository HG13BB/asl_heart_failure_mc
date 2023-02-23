# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Covertype Classifier trainer script."""
import os
import pickle
import subprocess
import sys
import io 
import fire
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    Dense, 
    Input
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.cloud import storage



#AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]
#MODEL_FILENAME = "model.pkl"


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
    contents = blob.download_as_text()

    return contents

def train_evaluate_tensorflow (dataset_uri, output_uri, batch_size=100, lr=.001, epochs=100 ):
    batch_size=int(batch_size)
    lr=float(lr)
    epochs=int(epochs)
    
    print("dataset_uri:",dataset_uri)
    print("output_uri:",output_uri)
    print("batch_size:",batch_size)
    print("lr:",lr)
    print("epochs:",epochs)
    
    
    bucket = dataset_uri.split("/")[2]
    input_object_name = "/".join(dataset_uri.split("/")[3:])
    
    
    input_string = download_blob_into_string(bucket, input_object_name)
    #print("input string:", input_string)
    dataset_df = pd.read_csv(io.StringIO(input_string), on_bad_lines="skip")

    print("dataset:", len(dataset_df))
    dataset_df.head()
    dataset_df.describe()
    
    train_df = dataset_df[dataset_df['split'] == "TRAIN"]
    val_df = dataset_df[dataset_df['split'] == "VALIDATE"]
    
    ytrain_df = train_df['HeartDisease']
    xtrain_df = train_df.drop(['split','HeartDisease'], axis=1)
    print("ytrain_df:", len(ytrain_df))
    ytrain_df.head()
    ytrain_df.describe()
    print("xtrain_df:", len(ytrain_df))
    xtrain_df.head()
    xtrain_df.describe()
    
    print("shape = ", xtrain_df.shape)
    
    
    yval_df = val_df['HeartDisease']
    xval_df = val_df.drop(['split','HeartDisease'], axis=1)
    
    dnn_hidden_units = [32, 16, 8]

    model = Sequential(name="dnn")
    
    model.add(Input(name="Input", shape=(22,)))
    
    for layer in dnn_hidden_units:
        model.add(Dense(units=layer, activation="relu", name=f"hidden_{layer}"))

    model.add(
        Dense(
            units=1,
            #activation="softmax",
            activation="sigmoid",
            #kernel_regularizer=tf.keras.regularizers.l1(l=0.1),
            name="Output",
        )
    )

    model.compile(
        optimizer=Adam(lr=lr),
        #loss="categorical_crossentropy",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(from_logits=True)],
    )
    
    history = model.fit(
        x=xtrain_df.values,
        y=ytrain_df,
        batch_size=batch_size,
        validation_data=(xval_df.values, yval_df),
        epochs=epochs,
        verbose=2,
    )
    
    model.summary()
    #print("\nsummary:\n", model.summary)
    print("\nhistory:\n", history.history)
    
    temp = model.evaluate(xval_df.values, yval_df)
    print("temp:", temp)
    
    #model.save(model_filename, overwrite=True, include_optimizer=True, save_format="tf", save_traces=True)
    tf.saved_model.save(model, export_dir=output_uri)
    
    #subprocess.check_call(
    #        ["gsutil", "cp", "saved_model.pb", output_uri], stderr=sys.stdout
    #    )
    print(f"Saved model in: {output_uri}")

def train_evaluate_sklearn (
    training_dataset_path, validation_dataset_path, alpha, max_iter, hptune
):
    """Trains the Covertype Classifier model."""

    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)

    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    numeric_feature_indexes = slice(0, 10)
    categorical_feature_indexes = slice(10, 12)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feature_indexes),
            ("cat", OneHotEncoder(), categorical_feature_indexes),
        ]
    )
    
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", SGDClassifier(loss="log")),
        ]
    )

    num_features_type_map = {
        feature: "float64"
        for feature in df_train.columns[numeric_feature_indexes]
    }
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)

    print(f"Starting training: alpha={alpha}, max_iter={max_iter}")
    # pylint: disable-next=invalid-name
    X_train = df_train.drop("Cover_Type", axis=1)
    y_train = df_train["Cover_Type"]

    pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
    pipeline.fit(X_train, y_train)

    if hptune:
        # pylint: disable-next=invalid-name
        X_validation = df_validation.drop("Cover_Type", axis=1)
        y_validation = df_validation["Cover_Type"]
        accuracy = pipeline.score(X_validation, y_validation)
        print(f"Model accuracy: {accuracy}")
        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="accuracy", metric_value=accuracy
        )

    # Save the model
    if not hptune:
        with open(MODEL_FILENAME, "wb") as model_file:
            pickle.dump(pipeline, model_file)
        subprocess.check_call(
            ["gsutil", "cp", MODEL_FILENAME, AIP_MODEL_DIR], stderr=sys.stdout
        )
        print(f"Saved model in: {AIP_MODEL_DIR}")


if __name__ == "__main__":
    fire.Fire(train_evaluate_tensorflow)
