import base64, json
from traceback import format_exc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, StandardScaler
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']



def get_input(argv, logging):
    try:   
        if len(argv) == 2:
            input_ = argv[1]
            input_ = base64.b64decode(input_).decode('utf-8')
            input_ = json.loads(input_)

            return input_
        else:
            logging.info("Input parameter error.")
    except:
        logging.error(format_exc())



def error(logging, message, model_id):
    logging.error(message)
    result = {
        "status": "fail",
        "reason": message,
        "model_id": model_id
        }
    
    return result



def generate_feature(df):
    df["預期_F側不平衡量"] = df["初始_F側不平衡量"] - df["平衡_F側配重"]
    df["預期_L側不平衡量"] = df["初始_L側不平衡量"] - df["平衡_L側配重"]

    df["初始_不平衡量_diff"] = abs(df["初始_L側不平衡量"] - df["初始_F側不平衡量"])
    df["平衡_不平衡量_diff"] = abs(df["平衡_L側配重"] - df["平衡_F側配重"])

    diff = abs(df["初始_L側角度"] - df["初始_F側角度"])
    df["初始_角度_diff"] = diff.apply(lambda X: min(X, 360 - X))
    diff = abs(df["平衡_L側角度"] - df["平衡_F側角度"])
    df["平衡_角度_diff"] = diff.apply(lambda X: min(X, 360 - X))
    
    return df



def split_data(df, test_size, shuffle, random_state):
    train, test = train_test_split(df, test_size = test_size, shuffle = shuffle, random_state = random_state)
    train = train.reset_index(drop = True)
    test  = test.reset_index(drop = True)

    return train, test



def deal_with_outlier(features, train, test):

    outlier_boundary = {}
    for col in features[1:]:
        Q1   = train[col].quantile(0.25)
        Q3   = train[col].quantile(0.75)
        IQR  = Q3 - Q1
        min_ = Q1 - (1.5 * IQR)
        max_ = Q3 + (1.5 * IQR)
        
        train[col] = train[col].apply(lambda X: max_ if X > max_ else X)
        train[col] = train[col].apply(lambda X: min_ if X < min_ else X)

        test[col]  = test[col].apply(lambda X: max_ if X > max_ else X)
        test[col]  = test[col].apply(lambda X: min_ if X < min_ else X)

        outlier_boundary[col] = {
            "min": min_,
            "max": max_,
        }

    return train, test, outlier_boundary



def deal_with_skew(features, train, test):
    skewness  = train[features].apply(lambda X: skew(X)).sort_values(ascending=False)
    skewness  = pd.DataFrame({'Feature' : skewness.index, 'Skew' : skewness.values})
    skewness  = skewness.query("(Skew > 0.75) | (Skew < -0.75)").reset_index(drop = True)
    skew_feat = skewness["Feature"].to_list()

    pt = PowerTransformer(method = 'yeo-johnson')
    train[skew_feat] = pt.fit_transform(train[skew_feat])
    test[skew_feat]  = pt.transform(test[skew_feat])

    return train, test, skew_feat, pt



def scaling(features, train, test):
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features]  = scaler.transform(test[features])

    return train, test, scaler