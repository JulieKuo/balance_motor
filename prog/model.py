import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']



def modeling(X_train, y_train, random_state = 99):
    models = {
        "Linear":         LinearRegression(),
        "Linear (L2)":    Ridge(random_state = random_state),
        "Linear (L1)":    Lasso(random_state = random_state),
        "Neural Network": MLPRegressor(random_state = random_state),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        # print(name + " trained.")
    
    return models



def calculate_score(models, X_train, X_test, y_train, y_test, scoring = "r2", cv_flag = 1, cv_scoring = "r2", random_state = 99):
    col = [("mse", "train"), ("mse", "test"), ("rmse", "train"), ("rmse", "test"), ("mape", "train"), ("mape", "test"), ("r2", "train"), ("r2", "test")]
    if cv_flag:
        col.append(((scoring, "cv")))
    col = pd.MultiIndex.from_tuples(col)

    pred_trains = pd.DataFrame(y_train.values, columns = ["target"])
    pred_tests  = pd.DataFrame(y_test.values, columns = ["target"])
    scores = pd.DataFrame()

    for name, model in models.items():
        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)

        pred_trains[name] = pred_train
        pred_tests[name]  = pred_test

        mse_train  = mean_squared_error(y_train, pred_train).round(2)
        mse_test   = mean_squared_error(y_test, pred_test).round(2)
        rmse_train = (mse_train ** (1/2)).round(2)
        rmse_test  = (mse_test ** (1/2)).round(2)
        mape_train = mean_absolute_percentage_error(y_train, pred_train).round(2)
        mape_test  = mean_absolute_percentage_error(y_test, pred_test).round(2)
        r2_train   = r2_score(y_train, pred_train).round(2)
        r2_test    = r2_score(y_test, pred_test).round(2)

        score = [mse_train, mse_test, rmse_train, rmse_test, mape_train, mape_test, r2_train, r2_test]

        if cv_flag:
            cv = ShuffleSplit(3, random_state = random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv = cv, scoring = cv_scoring)
            cv_score  = cv_scores.mean().round(2)
            score.append(cv_score)
        
        scores.loc[name, col] = score
    
    return scores, pred_trains, pred_tests



def pred_plot(pred_trains, pred_tests, scores, target_vol, model_path, side, top_score = 3):
    pred_trains_top = pred_trains[["target"] + scores.index[:top_score].to_list()]
    pred_tests_top  = pred_tests[["target"] + scores.index[:top_score].to_list()]

    data    = [pred_trains_top, pred_tests_top]
    titles  = [f"{side}側_train data", f"{side}側_test data"]
    fig, ax = plt.subplots(2, 1, figsize = (20, 10))
    for i in range(2):
        ax[i].plot(data[i], alpha = 1)
        ax[i].set(ylabel = target_vol[0], xlabel = "Sample", title = titles[i])
        ax[i].legend(data[i].columns, fontsize = 10)
    
    fig.savefig(f"{model_path}/{side}_pred.png")