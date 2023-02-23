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
import math
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
import tensorflow_datasets as tfds
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

LABEL_COLUMN="HeartDisease"
CSV_COLUMNS=['Sex.F', 'Sex.M', 'ChestPainType.ASY', 'ChestPainType.ATA', 'ChestPainType.NAP', 'ChestPainType.TA', 'FastingBS.0', 'FastingBS.1', 'RestingECG.LVH', 'RestingECG.Normal', 'RestingECG.ST', 'ExerciseAngina.N', 'ExerciseAngina.Y', 'ST_Slope.Down', 'ST_Slope.Flat', 'ST_Slope.Up', 'Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'binned_Oldpeak', LABEL_COLUMN]
DEFAULTS= [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

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

def features_and_labels(row_data):
    """Splits features and labels from feature dictionary.

    Args:
        row_data: Dictionary of CSV column names and tensor values.
    Returns:
        Dictionary of feature tensors and label tensor.
    """
    label = row_data.pop(LABEL_COLUMN)

    return row_data, label  # features, label    
    
def load_dataset(pattern, batch_size=1, mode=tf.estimator.ModeKeys.EVAL):
    """Loads dataset using the tf.data API from CSV files.

    Args:
        pattern: str, file pattern to glob into list of files.
        batch_size: int, the number of examples per batch.
        mode: tf.estimator.ModeKeys to determine if training or evaluating.
    Returns:
        `Dataset` object.
    """
    
    print("pattern = ", pattern)
    # Make a CSV dataset
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=pattern,
        batch_size=batch_size,
        column_names=CSV_COLUMNS,
        column_defaults=DEFAULTS,
    )

    # Map dataset to features and label
    dataset = dataset.map(map_func=features_and_labels)  # features, label

    # Shuffle and repeat for training
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=10).repeat()

    # Take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(buffer_size=1)
    return dataset

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
    
    features = list(dataset_df.columns)
    print("features = ", features)
    features = features[0:22]
    #features = features.remove('HeartDisease')
    print("features = ", features)

    print("dataset:", len(dataset_df))
    dataset_df.head()
    dataset_df.describe()
    
    train_df = dataset_df[dataset_df['split'] == "TRAIN"]
    val_df = dataset_df[dataset_df['split'] == "VALIDATE"]
    train_df = train_df.drop(['split'], axis=1)
    val_df = val_df.drop(['split'], axis=1)     
    
    train_csv_file = "./temp_train.csv"
    val_csv_file = "./temp_val.csv"
    train_df.to_csv(train_csv_file, index=False)
    val_df.to_csv(val_csv_file, index=False)
    
    ytrain_df = train_df[LABEL_COLUMN]
    xtrain_df = train_df.drop([LABEL_COLUMN], axis=1)
    print("ytrain_df:", len(ytrain_df))
    ytrain_df.head()
    ytrain_df.describe()
    print("xtrain_df:", len(ytrain_df))
    print("xtrain_df", xtrain_df.head())
    xtrain_df.head()
    xtrain_df.describe()
    
    print("shape = ", xtrain_df.shape)
    
    
    yval_df = val_df[LABEL_COLUMN]
    xval_df = val_df.drop([LABEL_COLUMN], axis=1)
    
    trainds = load_dataset(train_csv_file, batch_size, tf.estimator.ModeKeys.TRAIN)
    evalds = load_dataset(val_csv_file, batch_size, tf.estimator.ModeKeys.EVAL)
    #testds = load_dataset(test_csv_file, batch_size, tf.estimator.ModeKeys.EVAL)
    #print("trainds = ", tfds.as_dataframe(trainds.take(10)))
    #print("evalds = ", tfds.as_dataframe(evalds.take(10)))
    
    dnn_hidden_units = [32, 16, 8]


    
    
    #model.add(Input(name="Input", shape=(22,)))
    #numerical_inputs = {
    #    colname: tf.keras.layers.Input(
    #        name=colname, shape=(1,), dtype="float32"
    #    )
    #    for colname in numeric_cols
    #}
    
    
    #bucketized_inputs = {
    #    colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype="float32")
    #    for colname in bucketized_cols
    #}

    #categorial_inputs = {
    #    colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype="float32")
    #    for colname in categorical_cols
    #}
     
    #inputs = {**numerical_inputs, **bucketized_inputs, **categorial_inputs}
    
    inputs = {
        colname: tf.keras.layers.Input(name=colname, shape=(1,), dtype="float32")
        for colname in features
    }
    
    # The Functional API in Keras requires: LayerConstructor()(inputs)
    hidden_layer = tf.keras.layers.Concatenate()(inputs.values())
    
    #for layer in dnn_hidden_units:
    #    hidden_layer = Dense(units=layer, activation="relu", name=f"hidden_{layer}")(hidden_layer)

    hidden_layer = Dense(units=xtrain_df.shape[1], activation="relu", name=f"hidden")(hidden_layer)                    
                          
    output = Dense(
            units=1,
            #activation="softmax",
            activation="sigmoid",
            #kernel_regularizer=tf.keras.regularizers.l1(l=0.1),
            name="Output",
        )(hidden_layer)
                          
                        
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=Adam(lr=lr),
        #loss="categorical_crossentropy",
        loss="binary_crossentropy",
        #run_eagerly=True,
        metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(from_logits=True)],
    )
    
    #print("xtrain_df values type:", type(xtrain_df.values))
    #print("xtrain_df type:", type(xtrain_df))
    
    model.summary()
    tf.keras.utils.plot_model(model=model, to_file="simple_dnn_model.png", show_shapes=False, rankdir="LR")


    samples = train_df.shape[0]
    print("samples = ", samples)
    print("batch_size = ", batch_size)
    print("epochs = ", epochs)
    steps_per_epoch = math.ceil(samples/batch_size)
    print("steps_per_epoch = ", steps_per_epoch)
    print("trainds= ", trainds)
    print("evalds = ", evalds)
    print("trainds size:", trainds.cardinality(), trainds.__len__())
    print("evalds size:", evalds.cardinality(), evalds.__len__())
    
    history = model.fit(
        #x=xtrain_df.astype(float).to_dict(orient="records"),
        #y=ytrain_df,
        trainds,
        batch_size=batch_size,
        #validation_data=(xval_df.astype(float).to_dict(orient="records"), yval_df),
        steps_per_epoch=1,
        validation_data=evalds,
        epochs=epochs,
        verbose=2,
    )
    
    model.summary()
    #print("\nsummary:\n", model.summary)
    print("\nhistory:\n", history.history)
    
    #model.save(model_filename, overwrite=True, include_optimizer=True, save_format="tf", save_traces=True)
    #tf.saved_model.save(model, export_dir=AIP_MODEL_DIR)
    #tf.saved_model.save(model, export_dir="./output_model")
    model.save("./output_model") # for local save
    
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
