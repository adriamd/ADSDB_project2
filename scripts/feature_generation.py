import os
import duckdb
import pandas as pd
import numpy as np
try:
    from sandbox_generator import createSandbox
except:
    from scripts.sandbox_generator import createSandbox
import duckdb
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def log_transformation(df,cols=[]):
    name_columns = list(df.columns)
    types_columns = [str(x) for x in list(df.dtypes)]
    columnss = [(name_columns[i],types_columns[i]) for i in range(len(name_columns))]
    
    cols = set(cols)
    for i,t in columnss:

        if (t != "float64" and t != "int32") or i not in cols:
            continue
        
        df[i] = df[i].map(lambda x: np.log(x+1))

def outliers_cutpoints(df,col = "price",alpha=0.01):
  var = np.sort(df[col].to_numpy())
  var = var[~np.isnan(var)]
  n = var.size
  k = round(n*alpha/2)
  bot = k
  top = n-1-k

  return (var[bot],var[top])

def clean_outliers(df,cols=[]):
    name_columns = list(df.columns)
    types_columns = [str(x) for x in list(df.dtypes)]
    columnss = [(name_columns[i],types_columns[i]) for i in range(len(name_columns))]
    
    cols = set(cols)
    for i,t in columnss:

        if (t != "float64" and t != "int32") or i not in cols:
            continue
            
        (bot, top) = outliers_cutpoints(df,col = i,alpha=0.01)
        df[i] = df[i].map(lambda x: x if x >= bot and x<= top else np.nan)

def column_types(df):
    name_columns = list(df.columns)
    types_columns = [str(x) for x in list(df.dtypes)]
    
    columns = [(name_columns[i],types_columns[i]) for i in range(len(name_columns))]

    num_var = []
    cat_var = []

    for i,t in columns:
        if t == "object":
            cat_var.append(i)
        else:
            num_var.append(i)

    return (num_var, cat_var)

def numeric_imputation(df):
    df_num = df.select_dtypes(include=np.number)
    numeric_cols = df_num.columns

    lr = LinearRegression()
    imp = IterativeImputer(estimator=lr,missing_values=np.nan, max_iter=10, imputation_order='roman',random_state=0)
    df_num = pd.DataFrame(imp.fit_transform(df_num), columns=numeric_cols)

    for i in numeric_cols:
        #df.loc[:,i] = df_num.loc[:,i]
        df[i] = df_num[i].values

    return df

def new_variables(df):
    df["N_baths"] = df["baths"].apply(np.floor)
    df["half_baths"] =  (df["baths"] - df["N_baths"]).apply(lambda x:"Yes" if 0 < x else "No") 
    df["b_hospital_type_critical"] = df["hospital_type_critical"].apply(lambda x:"Yes" if 0 < x else "No")
    df["b_hospital_type_longterm"] = df["hospital_type_longterm"].apply(lambda x:"Yes" if 0 < x else "No")
    df["b_hospital_type_children"] = df["hospital_type_children"].apply(lambda x:"Yes" if 0 < x else "No")
    df["b_type"] = df["type"].apply(lambda x:"Apartment" if "apartment" == x else "No-Apartment")

    return df
    
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
            cols_encoded.append(str(col+'_'+name[0:].strip()).lower().strip())
    df_onehot_encoded = pd.DataFrame(enc.transform(df[x_cols_onehot]).toarray(), columns = cols_encoded)
    
    if scale:
        # scaling numerical variables:
        scaler = StandardScaler()
        df_scaled_num = pd.DataFrame(scaler.fit_transform(df[x_cols_numeric]), columns = x_cols_numeric)
        
        df.reset_index(drop=True, inplace=True)
        df_onehot_encoded.reset_index(drop=True, inplace=True)
        df_scaled_num.reset_index(drop=True, inplace=True)

        # dataset with encoding + scaling
        return [
            pd.concat(objs=[df[x_cols_binary], df_onehot_encoded, df_scaled_num, df[[y_col]]], axis = 1), 
            {'names':scaler.feature_names_in_, 'means':scaler.mean_, 'scales':scaler.scale_}
        ]
    
    # dataset with encoding categorical columns to numeric
    df.reset_index(drop=True, inplace=True)
    df_onehot_encoded.reset_index(drop=True, inplace=True)
    df_concat = pd.concat(objs=[
        df[x_cols_binary], df_onehot_encoded, df[x_cols_numeric], df[[y_col]]
    ], axis = 1
    )
    return df_concat

def split_train_test(dff, size_train = 0.7, random_seed=777, y_col='price'):
    df_train, df_test = np.split(
        dff.sample(frac=1, random_state=random_seed), [ int(size_train*len(dff)) ]
    )

    return [
        df_train.drop(y_col, axis=1), # X train
        df_train[y_col], # y train
        df_test.drop(y_col, axis=1), # X test
        df_test[y_col] # y test
    ]

def preprocessing(data, target, drop_features = [], skip_log=[], skip_outliers=[]):

    con = duckdb.connect('data/exploitation.db', read_only=False)
    df = con.execute(f"select * from {data}").df()
    df = df.drop(drop_features, axis=1)

    df = df.dropna(subset=[target])

    (num_var, cat_var) = column_types(df)

    num_cols = set(num_var) - set(skip_log)
    log_transformation(df,cols=num_cols)

    num_cols = set(num_var) - set(skip_outliers)
    clean_outliers(df,cols=num_cols)
    
    df = numeric_imputation(df)

    df = new_variables(df)

    df_encoded = encode(df, scale=False)
    df_scaled, scale_params = encode(df, scale=True)
    X_train, y_train, X_test, y_test = split_train_test(df_encoded)
    Xs_train, ys_train, Xs_test, ys_test = split_train_test(df_scaled)

    y_train = y_train.to_frame()
    y_test = y_test.to_frame()
    ys_train = ys_train.to_frame()
    ys_test = ys_test.to_frame()

    con.execute(f"CREATE OR REPLACE TABLE {data}_X_train_processed AS SELECT * FROM X_train")
    con.execute(f"CREATE OR REPLACE TABLE {data}_y_train_processed AS SELECT * FROM y_train")
    con.execute(f"CREATE OR REPLACE TABLE {data}_X_test_processed AS SELECT * FROM X_test")
    con.execute(f"CREATE OR REPLACE TABLE {data}_y_test_processed AS SELECT * FROM y_test")
    con.execute(f"CREATE OR REPLACE TABLE {data}_Xs_train_processed AS SELECT * FROM Xs_train")
    con.execute(f"CREATE OR REPLACE TABLE {data}_ys_train_processed AS SELECT * FROM ys_train")
    con.execute(f"CREATE OR REPLACE TABLE {data}_Xs_test_processed AS SELECT * FROM Xs_test")
    con.execute(f"CREATE OR REPLACE TABLE {data}_ys_test_processed AS SELECT * FROM ys_test")

    con.close()

    return df


if __name__ == "__main__":
    if os.getcwd().replace("\\", "/").split("/")[-1] in ["notebooks", "scripts"]:
        os.chdir("..")

    data = "sandbox_T_apartment_S_ca"
    
    df = preprocessing(data, target="price",
                    drop_features=["id", "url", "region_url", "image_url", "description"],
                    skip_log=["beds","baths","lat","long"],
                    skip_outliers=["lat","long"])

    




