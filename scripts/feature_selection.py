import duckdb
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from skfeature.function.information_theoretical_based.CIFE import cife


def feature_cife(df,target):

    X = df.drop(target, axis=1).to_numpy()
    y = df[target].to_numpy()

    name_columns = list(df.drop(target, axis=1).columns)

    (best_cife,_,_) = cife(X, y)
    print(best_cife)

    return [name_columns[i] for i in best_cife]


def feature_selection_random_forest(df,target):

    X = df.drop(target, axis=1).to_numpy()
    y = df[target].to_numpy()

    name_columns = list(df.drop(target, axis=1).columns)

    rf = RandomForestRegressor(oob_score=True,random_state=40).fit(X,y)
    threshold = np.mean(rf.feature_importances_)
    return [name_columns[i] for i in range(len(rf.feature_importances_)) if rf.feature_importances_[i] > threshold]


if __name__ == "__main__":
    table = "sandbox_T_apartment_S_ca"
    database = 'data/exploitation.db'
    target = "price"
    scaled = False

    con = duckdb.connect(database, read_only=True)

    if scaled:
        X_train = con.execute(f"select * from {table}_Xs_train_processed").df()
        y_train = con.execute(f"select * from {table}_ys_train_processed").df()
        X_test = con.execute(f"select * from {table}_Xs_test_processed").df()
    else:
        X_train = con.execute(f"select * from {table}_X_train_processed").df()
        y_train = con.execute(f"select * from {table}_y_train_processed").df()
        X_test = con.execute(f"select * from {table}_X_test_processed").df()
    con.close()

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    df = pd.concat(objs=[X_train, y_train ], axis = 1)
    
    df_sample = df.sample(round(len(df)*0.05), random_state=777)
    
    columns = feature_selection_random_forest(df_sample,target)
    
    X_train_reduced = X_train[columns]
    X_test_reduced = X_test[columns]

    con = duckdb.connect(database, read_only=False)

    if scaled:
        con.execute(f"CREATE OR REPLACE TABLE {table}_Xs_train_reduced AS SELECT * FROM X_train_reduced")
        con.execute(f"CREATE OR REPLACE TABLE {table}_Xs_test_reduced AS SELECT * FROM X_test_reduced")
    else:
        con.execute(f"CREATE OR REPLACE TABLE {table}_X_train_reduced AS SELECT * FROM X_train_reduced")
        con.execute(f"CREATE OR REPLACE TABLE {table}_X_test_reduced AS SELECT * FROM X_test_reduced")
    con.close()
