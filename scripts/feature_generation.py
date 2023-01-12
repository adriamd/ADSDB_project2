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

def new_variables(df):
    df["N_baths"] = df["baths"].apply(np.floor)
    df["half_baths"] =  (df["baths"] - df["N_baths"]).apply(lambda x:"Yes" if 0 < x else "No") 
    df["b_hospital_type_critical"] = df["hospital_type_critical"].apply(lambda x:"Yes" if 0 < x else "No")
    df["b_hospital_type_longterm"] = df["hospital_type_longterm"].apply(lambda x:"Yes" if 0 < x else "No")
    df["b_hospital_type_children"] = df["hospital_type_children"].apply(lambda x:"Yes" if 0 < x else "No")
    df["b_type"] = df["type"].apply(lambda x:"Apartment" if "apartment" == x else "No-Apartment")

def preprocessing(data, target, drop_features =[], skip_log=[], skip_outliers=[]):

    con = duckdb.connect('data/exploitation.db', read_only=False)
    df = con.execute(f"select * from {data}").df()
    df = df.drop(drop_features, axis=1)

    (num_var, cat_var) = column_types(df)

    num_cols = set(num_var) - set(skip_log)
    log_transformation(df,cols=num_cols)

    num_cols = set(num_var) - set(skip_outliers)
    clean_outliers(df,cols=num_cols)

    df = df.dropna(subset=[target])
    
    numeric_imputation(df)

    new_variables(df)

    con.execute(f"CREATE OR REPLACE TABLE {data}_preprocessed AS SELECT * FROM df")
    con.close()



if __name__ == "__main__":
    if os.getcwd().replace("\\", "/").split("/")[-1] in ["notebooks", "scripts"]:
        os.chdir("..")

    preprocessing(data="sandbox_T_apartment_S_ca", target="price",
                    drop_features =["id", "url", "region_url", "image_url", "description"],
                    skip_log=["beds","baths","lat","long"],
                    skip_outliers=["lat","long"])



