import duckdb
import pandas as pd
import numpy as np
from sandbox_generator import createSandbox
from time import time

import duckdb
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from datetime import datetime as dt
import os
from joblib import dump
import json

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

from skfeature.function.information_theoretical_based.CIFE import cife
# ENCODING

def encode(df, scale=False):
    # select columns
    y_col = 'price'
    exclude_cols = ['region']
    x_cols_onehot = ['type', 'state', 'laundry_options', 'parking_options']
    x_cols_binary = ['cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished', 'half_baths'] + [
        c for c in df.columns if c[0:2]=='b_'
    ]
    x_cols_numeric = [c for c in df.columns if not c in [y_col] + exclude_cols + x_cols_onehot + x_cols_binary]

    # binary to numeric
    bin2num = lambda x: 1 if x=='Yes' or x==1 or x=="Apartment" else 0
    for col in x_cols_binary:
        df[col] = df[col].apply(bin2num)

    # one-hot encoding in categorical variables
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df[x_cols_onehot])
    cols_encoded = []
    for i in range(len(x_cols_onehot)):
        col = x_cols_onehot[i]
        for name in enc.categories_[i]:
            cols_encoded.append(str(col+' '+name[0:].strip()).lower().strip())
    df_onehot_encoded = pd.DataFrame(enc.transform(df[x_cols_onehot]).toarray(), columns = cols_encoded)

    if scale:
        # scaling numerical variables:
        scaler = StandardScaler()
        df_scaled_num = pd.DataFrame(scaler.fit_transform(df[x_cols_numeric]), columns = x_cols_numeric)

        # dataset with encoding + scaling
        return [
            pd.concat(objs=[df[x_cols_binary], df_onehot_encoded, df_scaled_num, df[[y_col]]], axis = 1), 
            {'names':scaler.feature_names_in_, 'means':scaler.mean_, 'scales':scaler.scale_}
        ]

    # dataset with encoding categorical columns to numeric
    return pd.concat(objs=[
        df[x_cols_binary], df_onehot_encoded, df[x_cols_numeric], df[[y_col]]
    ], axis = 1
    )

def feature_cife(df,target):

    X = df.drop(target, axis=1).to_numpy()
    y = df[target].to_numpy()

    name_columns = list(df.drop(target, axis=1).columns)

    (best_cife,_,_) = cife(X, y)

    return [name_columns[i] for i in best_cife]


def feature_selection_random_forest(df,target):

    X = df.drop(target, axis=1).to_numpy()
    y = df[target].to_numpy()

    name_columns = list(df.drop(target, axis=1).columns)

    rf = RandomForestRegressor(oob_score=True,random_state=40).fit(X,y)
    threshold = np.mean(rf.feature_importances_)
    return [name_columns[i] for i in range(len(rf.feature_importances_)) if rf.feature_importances_[i] > threshold]


if __name__ == "__main__":
    table = "sandbox_T_apartment_S_ca_preprocessed"
    database = 'data/exploitation.db'
    target = "price"

    con = duckdb.connect(database, read_only=True)
    df = con.execute(f"select * from {table}").df()
    con.close()
    
    df_encoded = encode(df, scale=False)
    df_sample = df_encoded.sample(round(len(df)*0.1), random_state=777)
    
    columns = feature_cife(df_sample,target)
    columns.append(target)
    
    df_reduced = df_encoded[columns]
    con = duckdb.connect(database, read_only=False)
    out_table = table[:-13]
    con.execute(f"CREATE OR REPLACE TABLE {out_table}_reduced AS SELECT * FROM df_reduced")
    con.close()
