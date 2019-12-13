import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

import pandas as pd
import seaborn as sns


def read_data():
    train = pd.read_csv('data/exoTrain.csv')
    test = pd.read_csv('data/exoTest.csv')
    
    train.columns = [train.columns[i].replace("FLUX.", "") for i in range(len(train.columns))]
    test.columns = [test.columns[i].replace("FLUX.", "") for i in range(len(test.columns))]
    
    # Also replace 1 with 0 and then 2 with 1 in the label, as that makes more sense
    train.LABEL = train.LABEL.replace(1, 0)
    train.LABEL = train.LABEL.replace(2, 1)
    
    test.LABEL = test.LABEL.replace(1, 0)
    test.LABEL = test.LABEL.replace(2, 1)
    
    return (train, test)

def smoothen(df):
    labels = df.LABEL
    df = df.drop(["LABEL"], axis=1)
    
    for i in range(len(df)):
        df.iloc[i] = ndimage.filters.gaussian_filter(df.iloc[1], sigma=15)
        
    df["LABEL"] = labels

def normalize_bound(df):
    labels = df.LABEL
    df = df.drop(["LABEL"], axis=1)
    
    norm = lambda row : (row - np.mean(row)) / (np.max(row) - np.min(row))
    for i in range(len(df)):
        df.iloc[i] = norm(df.iloc[i])
    
    df["LABEL"] = labels

def normalize_deviation(df):
    labels = df.LABEL
    df = df.drop(["LABEL"], axis=1)
    
    norm = lambda row : (row - np.mean(row)) / (np.std(row))
    for i in range(len(df)):
        df.iloc[i] = norm(df.iloc[i])
    
    df["LABEL"] = labels