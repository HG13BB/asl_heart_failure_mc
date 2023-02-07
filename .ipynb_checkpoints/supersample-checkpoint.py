
import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from tensorflow import feature_column as fc
from tensorflow.keras import layers
from IPython.display import display, HTML
import seaborn as sns

print("TensorFlow version: ", tf.version.VERSION)


hf_df = pd.read_csv(F'gs://qwiklabs-asl-02-99f66d8df225/heart_failure/heart.csv')



def oversample_df(df,addsamples,test_size,random_state):
    #oversample the dataframe to get to ~1000 samples for automl.. keep dupes in same split
    
    NUM_ROWS_TO_DUPLICATE = addsamples
    TRAIN_SPLIT=1-test_size
    VAL_SPLIT=test_size/2
    TEST_SPLIT=test_size/2
    RANDOM_SEED=42
    
    Y_Values = df["HeartDisease"]

    train, test = train_test_split(df, random_state=RANDOM_SEED, test_size=(TEST_SPLIT - TEST_SPLIT*VAL_SPLIT), stratify=Y_Values)

    Y_Values = train["HeartDisease"]
    
    train, val = train_test_split(train, random_state=RANDOM_SEED*2, test_size=VAL_SPLIT, stratify=Y_Values)

    train['split'] = "TRAIN"
    val['split'] = "VALIDATE"
    test['split'] = "TEST"
    
    train_scaled = pd.concat([train, train.sample(n=math.ceil(NUM_ROWS_TO_DUPLICATE*TRAIN_SPLIT), random_state=RANDOM_SEED)],axis=0)
    val_scaled = pd.concat([val, val.sample(n=math.ceil(NUM_ROWS_TO_DUPLICATE*VAL_SPLIT), random_state=RANDOM_SEED)],axis=0)
    test_scaled = pd.concat([test, test.sample(n=math.ceil(NUM_ROWS_TO_DUPLICATE*TEST_SPLIT), random_state=RANDOM_SEED)],axis=0)
    



    scaled_dataset = pd.concat([train_scaled,val_scaled, test_scaled], axis=0)
    
    return scaled_dataset
    
hd_df_oversample = oversample_df(hf_df,addsamples=100,test_size=.2,random_state=42)


#split to three files to use the tf.data workflow

split = hd_df_oversample.split.unique()

col = list(hd_df_oversample.columns)
keep_col = col.pop(col.index('split'))



for s in split:
    hd_df_oversample[hd_df_oversample['split']== s][hd_df_oversample.columns[:-1]].to_csv(F'gs://qwiklabs-asl-02-99f66d8df225/heart_failure/heart_failure_' + s + '.csv',index=False)


    




