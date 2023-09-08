import base64, json, random
from traceback import format_exc
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, StandardScaler
import pandas as pd
import numpy as np
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



def generate_weight(weight_ori, gap, count, weight_limit):
    end = round((count + 1) * gap) # 候選重量的結束範圍
    weight_ori = min(weight_ori, weight_limit) # 重量不可大於上限
    
    # 產生候選重量
    weight_change = []
    for i in range(0, end, gap):
        weight_change.extend([max(0, weight_ori + i), max(0, weight_ori - i)])

    # 刪除相同的值
    weight_change = sorted(list(set(weight_change)))
    weight_change = np.array(weight_change)

    # 重量不可大於上限
    weight_change = weight_change[weight_change <= weight_limit]

    return weight_change



def generate_angle(angle_ori, material, aluminum_division, copper_division, max_k):
    # 產生候選角度
    if material == "aluminum":
        angle_init = np.linspace(0, 360, (aluminum_division + 1))
        k = min(round(90 / (360 / aluminum_division)), max_k) # 候選組合只可在同象限內
    else:    
        angle_init = np.linspace(0, 360, (copper_division + 1))
        k = min(round(90 / (360 / copper_division)), max_k) # 候選組合只可在同象限內

    closest_index = np.argsort(np.abs(angle_init - angle_ori))[:k] # 與angle_ori最接近的K個角度
    angle_change0 = angle_init[closest_index]


    # 調整候選較度
    angle_change = []
    for angle in angle_change0:
        if angle > 360:
            angle_change.append(angle - 360)
        elif angle < 0:
            angle_change.append(360 + angle)
        else:
            angle_change.append(angle)
    
    # 排序候選角度
    angle_change = sorted(angle_change)

    return angle_change, angle_init



def remove_combination(df_db, angle_init, weight_limit, df_comb, side):
    # db中的實際填補角度以最接近的欲填補角度替代
    for i in range(len(df_db)):
        closest_index = np.argmin(np.abs(angle_init - df_db.loc[i, f"{side.lower()}_angle_act"]))
        df_db.loc[i, f"{side.lower()}_angle_act"] = angle_init[closest_index]

    # 計算已填補角度的剩餘可填補重量
    cumulative_weight = df_db.groupby(f"{side.lower()}_angle_act").agg({f"{side.lower()}_weight_act": sum})
    cumulative_weight = cumulative_weight.to_dict()[f"{side.lower()}_weight_act"]

    # 刪除超出可填補重量的候選組合
    for key, value in cumulative_weight.items():
        if value > 0:
            remain_min = 0
            remain_max = weight_limit - value
        else:
            remain_min = (- weight_limit) - value
            remain_max = 0

        drop_index = df_comb.query(f"平衡_{side}側角度 == {key} and (平衡_{side}側配重 > {remain_min}) and (平衡_{side}側配重 < {remain_max})").index
        df_comb = df_comb.drop(drop_index).reset_index(drop = True)
    
    return df_comb



def split_data(df):
    # 切分
    test_work_id = random.choices(df["工號"].unique(), k = round(df["工號"].nunique() * 0.2))
    test = df.query("工號 in @test_work_id")
    train = df.query("工號 not in @test_work_id")

    # 按工號插分複製資料
    train1 = pd.DataFrame()
    g = train.groupby("工號")
    for group in train["工號"].unique():
        # 儲存原始資料      
        df_group0 = g.get_group(group)
        df_group = df_group0.loc[:, "平衡轉速":]
        train1 = pd.concat([train1, df_group], ignore_index = True)

        
        df_diff = df_group.shift(-1) - df_group # 兩個點之間的距離，即插分範圍
        times = int(np.ceil(1200 / len(train))) + 4 # 插分次數
        for _ in range(times):
            df_diff1 = df_diff / 10 * random.uniform(1, 9) # 插分的距離落點
            df_group1 = df_group + df_diff1 # 原數據 + 插分的距離落點 = 新數據點
            df_group1["平衡轉速"] = df_group0["平衡轉速"] # 使用前一個值
            df_group1 = df_group1.dropna()

            train1 = pd.concat([train1, df_group1], ignore_index = True)

    # 角度取整數，平衡值取小數點第二位
    angle_col = [col for col in df.columns if ("角度" in col)]
    train1[angle_col] = train1[angle_col].round().astype(int)
    train1 = train1.round(2)

    # 角度和平衡值不可為負
    check_col = [col for col in df.columns if ("初始" in col) or ("最終" in col)]
    train1 = train1[(train1[check_col] >= 0).all(axis = 1)]

    # 刪除重複的數據點
    train1 = train1.drop_duplicates(keep = "first").reset_index(drop = True)
    train = train1.copy()
    test = test[train.columns]

    return train, test, test_work_id



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