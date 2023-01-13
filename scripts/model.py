import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from datetime import datetime as dt
import os
from joblib import dump
import xgboost as xgb
import json

# ONE FUNCTION FOR EACH ALGORITHM

def printMetrics(y_true, y_pred):
    print("RMSE: ", round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 4))
    print("MAE: ", round(metrics.mean_absolute_error(y_true, y_pred),4))
    print("MAPE: ", round(metrics.mean_absolute_percentage_error(y_true, y_pred),4))
    print("R2: ", round(metrics.r2_score(y_true, y_pred),4))

def scatterplots(y_tr, yhat_tr, y_te, yhat_te, model_name):
    plt.scatter(y_tr, yhat_tr)
    plt.title("Training set")
    plt.xlabel("real value")
    plt.ylabel("predicted value")
    plt.savefig("output/"+model_name+"/plot_train.png")
    plt.clf()
    plt.scatter(y_te, yhat_te)
    plt.title("Test set")
    plt.xlabel("real value")
    plt.ylabel("predicted value")
    plt.savefig("output/"+model_name+"/plot_test.png")
    plt.clf()


def linear_regression(X_train, y_train, X_test, y_test, save_plots=True):
    model_name = "linreg_" + dt.now().strftime("%Y%m%d_%H%M%S")

    reg = LinearRegression().fit(X_train, y_train)
    yhat_train_lin = reg.predict(X_train)
    yhat_val_lin = reg.predict(X_test)

    print("Metrics in train set:")
    printMetrics(y_train, yhat_train_lin)
    print("Metrics in test set:")
    printMetrics(y_test, yhat_val_lin)

    # save the model
    os.mkdir("output/"+model_name)
    dump(reg, "output/"+model_name+"/linreg.joblib")

    # save the plots
    if save_plots:
        scatterplots(y_train, yhat_train_lin, y_test, yhat_val_lin, model_name)

    return reg

# Ridge regression, with loo-cv to choose the regularization parameter
def ridge_regression(X_train, y_train, X_test, y_test, save_plots=True):
    model_name = "ridgereg_" + dt.now().strftime("%Y%m%d_%H%M%S")

    rreg = RidgeCV(alphas = np.logspace(-3,3,7)).fit(X_train, y_train)
    yhat_train_rr = rreg.predict(X_train)
    yhat_val_rr = rreg.predict(X_test)

    print("Chosen alpha parameter: ", rreg.alpha_)
    print("Metrics in train set:")
    printMetrics(y_train, yhat_train_rr)
    print("Metrics in test set:")
    printMetrics(y_test, yhat_val_rr)

    os.mkdir("output/"+model_name)
    dump(rreg, "output/"+model_name+"/ridgereg.joblib")

    if save_plots:
        scatterplots(y_train, yhat_train_rr, y_test, yhat_val_rr, model_name)

    return rreg


# Lasso regression, with loo-cv to choose the regularization parameter
def lasso_regression(X_train, y_train, X_test, y_test, save_plots=True):
    model_name = "lassoreg_" + dt.now().strftime("%Y%m%d_%H%M%S")

    lreg = LassoCV(alphas = np.logspace(-3,3,7)).fit(X_train, y_train)
    yhat_train_lr = lreg.predict(X_train)
    yhat_val_lr = lreg.predict(X_test)

    print("Chosen alpha parameter: ", lreg.alpha_)
    print("Metrics in train set:")
    printMetrics(y_train, yhat_train_lr)
    print("Metrics in test set:")
    printMetrics(y_test, yhat_val_lr)

    os.mkdir("output/"+model_name)
    dump(lreg, "output/"+model_name+"/lassoreg.joblib")

    if save_plots:
        scatterplots(y_train, yhat_train_lr, y_test, yhat_val_lr, model_name)

    return lreg

def random_forest(X_train, y_train, X_test, y_test, save_plots=True):
    model_name = "randomforest_" + dt.now().strftime("%Y%m%d_%H%M%S")

    rf = RandomForestRegressor(oob_score=True).fit(X_train, y_train)
    yhat_train_rf = rf.predict(X_train)
    yhat_val_rf = rf.predict(X_test)

    print("Metrics in train set:")
    printMetrics(y_train, yhat_train_rf)
    print("Metrics in test set:")
    printMetrics(y_test, yhat_val_rf)

    os.mkdir("output/"+model_name)
    dump(rf, "output/"+model_name+"/randomforest.joblib")

    if save_plots:
        scatterplots(y_train, yhat_train_rf, y_test, yhat_val_rf, model_name)
        # top 10 feature importance
        sorted_idx = rf.feature_importances_.argsort()
        plt.figure(figsize=(10,10))
        plt.barh(rf.feature_names_in_[sorted_idx][-20:], rf.feature_importances_[sorted_idx][-20:])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig("output/"+model_name+"/feature_importance_top20.png")
        plt.clf()

    return rf

# XGBoost, without crossvalidation. Uses subsampling to avoid overfitting
def xgboost_noCV(X_train, y_train, X_test, y_test, save_plots=True,
    params = {
        'objective':'reg:squarederror',
        'verbosity':1,
        'learning_rate':0.3,
        'max_depth':10,
        'subsample':0.8, # to avoid overfitting
        'reg_lambda':0.1, # L2 reg
        'alpha':0, #L1 reg
        'max_leaves':10
    }, n_estim=20):

    model_name = "xgb_" + dt.now().strftime("%Y%m%d_%H%M%S")

    data_train = xgb.DMatrix(X_train, label=y_train)
    data_val = xgb.DMatrix(X_test, label=y_test)
    xgb_1 = xgb.train(params, data_train, n_estim)
    yhat_train_xgb1 = xgb_1.predict(data_train)
    yhat_val_xgb1 = xgb_1.predict(data_val)

    print("Metrics in train set:")
    printMetrics(y_train, yhat_train_xgb1)
    print("Metrics in test set:")
    printMetrics(y_test, yhat_val_xgb1)

    os.mkdir("output/"+model_name)
    dump(xgb_1, "output/"+model_name+"/xgb.joblib")
    with open("output/"+model_name+"/xgb_params.json", "w") as fp:
        params['n_estim'] = n_estim
        json.dump(params, fp)

    if save_plots:
        scatterplots(y_train, yhat_train_xgb1, y_test, yhat_val_xgb1, model_name)
        xgb.plot_importance(xgb_1, max_num_features = 20, height = 1)
        fig = plt.gcf()
        fig.savefig("output/"+model_name+"/feature_importance_top20.png")
        plt.clf()
        try:
            xgb.plot_tree(xgb_1, num_trees = 4, rankdir="LR")
            fig = plt.gcf()
            fig.savefig("output/"+model_name+"/tree.png")
            plt.clf()
        except:
            pass
    
    return xgb_1

def xgboost_CV(X_train, y_train, X_test, y_test, save_plots=True, params = {
        'learning_rate':[0.1, 0.3, 1],
        'max_depth':[5, 7, 10],
        'subsample':[0.8],
        'reg_lambda':[0, 0.1, 1], # L2 reg
        'alpha':[0, 0.1, 1], #L1 reg
        'max_leaves':[10],
        'n_estimators':[5, 10, 20]
    }):

    print("This may take a few minutes...")
    model_name = "xgbcv_" + dt.now().strftime("%Y%m%d_%H%M%S")

    data_train = xgb.DMatrix(X_train, label=y_train)
    data_val = xgb.DMatrix(X_test, label=y_test)
    xgbr = xgb.XGBRegressor(seed = 20, objective = 'reg:squarederror', verbosity = 1)

    xgb_grid = GridSearchCV(estimator=xgbr, param_grid = params, cv = 3)
    xgb_grid.fit(X_train, y_train)
    # retrain with the best parameters:
    xgb_cv = xgb.train(xgb_grid.best_params_, data_train, xgb_grid.best_params_['n_estimators'])
    # prediction:
    yhat_train_xgbcv = xgb_cv.predict(data_train)
    yhat_val_xgbcv = xgb_cv.predict(data_val)

    print("Metrics in train set:")
    printMetrics(y_train, yhat_train_xgbcv)
    print("Metrics in test set:")
    printMetrics(y_test, yhat_val_xgbcv)

    os.mkdir("output/"+model_name)
    dump(xgb_grid, "output/"+model_name+"/xgb_grid.joblib")
    dump(xgb_cv, "output/"+model_name+"/xgb_cv.joblib")
    with open("output/"+model_name+"/grid_params.json", "w") as fp:
        json.dump(params, fp)
    with open("output/"+model_name+"/best_params.json", "w") as fp:
        json.dump(xgb_grid.best_params_, fp)

    if save_plots:
        scatterplots(y_train, yhat_train_xgbcv, y_test, yhat_val_xgbcv, model_name)
        xgb.plot_importance(xgb_cv, max_num_features = 20, height = 1)
        fig = plt.gcf()
        fig.savefig("output/"+model_name+"/feature_importance_top20.png")
        plt.clf()
        try:
            xgb.plot_tree(xgb_cv, num_trees = 4, rankdir="LR")
            fig = plt.gcf()
            fig.savefig("output/"+model_name+"/tree.png")
            plt.clf()
        except:
            pass

    return xgb_cv

def best_model(table, database_path = 'data/exploitation.db'):
    print("Running random forest:")

    con = duckdb.connect(database_path, read_only=True)
    df = con.execute(f"select * from {table}").df()
    con.close()

    df_encoded = encode(df, scale=False)
    X_train, y_train, X_test, y_test = split_train_test(df_encoded)
    random_forest(X_train, y_train, X_test, y_test)

def show_all_models(table, database_path = 'data/exploitation.db'):
    con = duckdb.connect(database_path, read_only=True)
    df = con.execute(f"select * from {table}").df()
    con.close()

    df_encoded = encode(df, scale=False)
    df_scaled, scale_params = encode(df, scale=True)
    # save the scale params in a file?

    X_train, y_train, X_test, y_test = split_train_test(df_encoded)
    Xs_train, ys_train, Xs_test, ys_test = split_train_test(df_scaled)

    x=1
    while x != "0":
        x = input("\nCreate model:\n[1] Linear regression\n[2] Ridge regression\n[3] Lasso regression\n[4] Random forest\n[5] XGB 1\n[6] XGB 2\n[0] Exit\n")
        if x=="1":
            linear_regression(X_train, y_train, X_test, y_test)
        elif x=="2":
            ridge_regression(Xs_train, ys_train, Xs_test, ys_test)
        elif x=="3":
            lasso_regression(Xs_train, ys_train, Xs_test, ys_test)
        elif x=="4":
            random_forest(X_train, y_train, X_test, y_test)
        elif x=="5":
            xgboost_noCV(X_train, y_train, X_test, y_test)
        elif x=="6":
            xgboost_CV(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    if os.getcwd().replace("\\", "/").split("/")[-1] in ["notebooks", "scripts"]:
        os.chdir("..")

    table = "sandbox_T_apartment_S_ca"
    database = 'data/exploitation.db'

    con = duckdb.connect(database, read_only=True)
    X_train = con.execute(f"select * from {table}_X_train_processed").df()
    y_train = con.execute(f"select * from {table}_y_train_processed").df()
    X_test = con.execute(f"select * from {table}_X_test_processed").df()
    y_test = con.execute(f"select * from {table}_y_test_processed").df()

    Xs_train = con.execute(f"select * from {table}_Xs_train_processed").df()
    ys_train = con.execute(f"select * from {table}_ys_train_processed").df()
    Xs_test = con.execute(f"select * from {table}_Xs_test_processed").df()
    ys_test = con.execute(f"select * from {table}_ys_test_processed").df()

    con.close()


    x=1
    while x != "0":
        x = input("\nCreate model:\n[1] Linear regression\n[2] Ridge regression\n[3] Lasso regression\n[4] Random forest\n[5] XGB 1\n[6] XGB 2\n[0] Exit\n")
        if x=="1":
            linear_regression(X_train, y_train, X_test, y_test)
        elif x=="2":
            ridge_regression(Xs_train, ys_train, Xs_test, ys_test)
        elif x=="3":
            lasso_regression(Xs_train, ys_train, Xs_test, ys_test)
        elif x=="4":
            random_forest(X_train, y_train, X_test, y_test)
        elif x=="5":
            xgboost_noCV(X_train, y_train, X_test, y_test)
        elif x=="6":
            xgboost_CV(X_train, y_train, X_test, y_test)
    