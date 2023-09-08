import os, sys, json, pickle, sqlalchemy
from traceback import format_exc
from log_config import Log
from tools import *



class Model():
    def __init__(self, root, input_, logging):
        self.logging = logging


        # 取得input參數
        self.get_input(input_)


        # 取得predict位置
        pred_path = os.path.join(root, "data", "predict")        
        os.makedirs(pred_path, exist_ok = True)
        self.output_json = os.path.join(pred_path, "output.json")


        # 取得model位置     
        self.model_detail = os.path.join(root, "data", "train", self.model_id, "model")


        # 取得config
        config_path = os.path.join(root, "prog", "config.json")
        with open(config_path) as f:
            self.config = json.load(f)



    def get_input(self, input_):
        self.work_id      = input_["work_id"]
        self.op           = input_["op"]
        self.model_id     = input_["model_id"]
        self.speed        = int(input_["speed"])
        self.l_angle_ori  = int(input_["l_angle_ori"])
        self.l_weight_ori = round(float(input_["l_weight_ori"]))
        self.f_angle_ori  = int(input_["f_angle_ori"])
        self.f_weight_ori = round(float(input_["f_weight_ori"]))
        self.material     = input_["material"]



    def load_model(self, side):
        features         = pickle.load(open(os.path.join(self.model_detail, f"feat_order.pkl"), "rb"))
        outlier_boundary = pickle.load(open(os.path.join(self.model_detail, f"{side}_outlier_boundary.pkl"), "rb"))
        skew_feat        = pickle.load(open(os.path.join(self.model_detail, f"{side}_skew_feat.pkl"), "rb"))
        pt               = pickle.load(open(os.path.join(self.model_detail, f"{side}_power_tf.pkl"), "rb"))
        scaler           = pickle.load(open(os.path.join(self.model_detail, f"{side}_scaler.pkl"), "rb"))
        model            = pickle.load(open(os.path.join(self.model_detail, f"{side}_model.pkl"), "rb"))


        return features, outlier_boundary, skew_feat, pt, scaler, model



    def get_db_data(self):
        con_info = f'mysql+pymysql://{self.config["user"]}:{self.config["password"]}@{self.config["host"]}:{self.config["port"]}/{self.config["database"]}'
        conn = sqlalchemy.create_engine(con_info)

        query = f'SELECT * FROM {self.config["table"]} WHERE (work_id = "{self.work_id}")'
        df_db = pd.read_sql(query, conn).sort_values("op")

        return df_db
        
    
    
    def predict(self, count = 20, gap = 1, aluminum_division = 12, copper_division = 72, copper_limit = 300, aluminum_limit = 60, max_k = 9):
        try:
            self.logging.info(f'Load model from {self.model_detail}\*.pkl')
            features, l_outlier_boundary, l_skew_feat, l_pt, l_scaler, l_model = self.load_model(side = "L")
            features, f_outlier_boundary, f_skew_feat, f_pt, f_scaler, f_model = self.load_model(side = "F")


            self.logging.info("Get DB data.")
            df_db = self.get_db_data()

            # 可填補角度要與起始角度同範圍，候選角度以初始角度為矛點
            if not df_db.empty:
                self.l_angle_ori, self.f_angle_ori = df_db.loc[0, "l_angle_ori"], df_db.loc[0, "f_angle_ori"]

            
            self.logging.info("Generate candidate values.")
            # 產生同向候選重量
            weight_limit = aluminum_limit if self.material == "aluminum" else copper_limit
            l_weight_change = generate_weight(self.l_weight_ori, gap, count, weight_limit)
            f_weight_change = generate_weight(self.f_weight_ori, gap, count, weight_limit)

            # 產生同向候選角度
            l_angle_change, l_angle_init = generate_angle(self.l_angle_ori, self.material, aluminum_division, copper_division, max_k)
            f_angle_change, f_angle_init = generate_angle(self.f_angle_ori, self.material, aluminum_division, copper_division, max_k)

            # 產生對向候選重量
            l_weight_change_opposite = l_weight_change * -1
            f_weight_change_opposite = f_weight_change * -1

            # 產生對向候選角度
            l_angle_change_opposite = [(angle - 180) if angle > 180 else (angle + 180) for angle in l_angle_change]
            f_angle_change_opposite = [(angle - 180) if angle > 180 else (angle + 180) for angle in f_angle_change]
            
            
            self.logging.info("Generate candidate combinations.")
            all_combinations = []
            for a in l_angle_change:
                for b in l_weight_change:
                    for c in f_angle_change:
                        for d in f_weight_change:
                            x = [self.speed, self.l_angle_ori, self.l_weight_ori, self.f_angle_ori, self.f_weight_ori, a, b, c, d]
                            all_combinations.append(x)
                            
            for a in l_angle_change_opposite :
                for b in l_weight_change_opposite:
                    for c in f_angle_change_opposite:
                        for d in f_weight_change_opposite:
                            x = [self.speed, self.l_angle_ori, self.l_weight_ori, self.f_angle_ori, self.f_weight_ori, a, b, c, d]
                            all_combinations.append(x)

            # self.logging.info(f"l_angle_change = {l_angle_change}")
            # self.logging.info(f"l_weight_change = {l_weight_change}")
            # self.logging.info(f"f_angle_change = {f_angle_change}")
            # self.logging.info(f"f_weight_change = {f_weight_change}")
            

            df_comb = pd.DataFrame(
                all_combinations, 
                columns = ['平衡轉速', '初始_L側角度', '初始_L側不平衡量', '初始_F側角度', '初始_F側不平衡量', '平衡_L側角度', '平衡_L側配重', '平衡_F側角度', '平衡_F側配重']
                )
            

            self.logging.info("Remove combinations.")
            if not df_db.empty:
                df_db = df_db.fillna(0)
                df_comb = remove_combination(df_db, l_angle_init, weight_limit, df_comb, side = "L")
                df_comb = remove_combination(df_db, f_angle_init, weight_limit, df_comb, side = "F")
            df_pred = df_comb.copy()


            self.logging.info("Generate features.")
            df_pred = generate_feature(df_pred)
            l_df, f_df = df_pred.copy(), df_pred.copy()


            self.logging.info("Feature engineering.")
            for col in features[1:]:    
                max_ = l_outlier_boundary[col]["max"]
                max_index = l_df.index[l_df[col] > max_].to_list()
                l_df.loc[max_index, col] = max_

                min_ = l_outlier_boundary[col]["min"]
                min_index = l_df.index[l_df[col] < min_].to_list()
                l_df.loc[min_index, col] = min_

                max_ = f_outlier_boundary[col]["max"]
                max_index = f_df.index[f_df[col] > max_].to_list()
                f_df.loc[max_index, col] = max_

                min_ = f_outlier_boundary[col]["min"]
                min_index = f_df.index[f_df[col] < min_].to_list()
                f_df.loc[min_index, col] = min_            

            l_df[l_skew_feat] = l_pt.transform(l_df[l_skew_feat])
            f_df[f_skew_feat] = f_pt.transform(f_df[f_skew_feat])
            l_df[features]    = l_scaler.transform(l_df[features])
            f_df[features]    = f_scaler.transform(f_df[features])


            self.logging.info("Predict.")
            pred_l = l_model.predict(l_df)
            pred_f = f_model.predict(f_df)

            df_comb["最終_L側不平衡量"] = pred_l
            df_comb["最終_F側不平衡量"] = pred_f


            self.logging.info("Find best combination.")
            df_comb["初始_總不平衡量"] = df_comb.eval("(abs(初始_L側不平衡量) + abs(初始_F側不平衡量))")
            df_comb["最終_總不平衡量"] = df_comb.eval("(abs(最終_L側不平衡量) + abs(最終_F側不平衡量))")

            # 正常不可能低於0.1
            df_comb1 = df_comb.query("(最終_L側不平衡量 >= 0.1) and (最終_F側不平衡量 >= 0.1)")
            if not df_comb1.empty:
                df_comb = df_comb1.reset_index(drop = True)

            # 先嚴格篩選，再寬鬆篩選
            df_comb2 = df_comb.query("((平衡_L側配重 > 0) and (平衡_F側配重 > 0)) and (最終_總不平衡量 < 初始_總不平衡量)")
            if df_comb2.empty:
                df_comb2 = df_comb.query("(最終_總不平衡量 < 初始_總不平衡量)")
            if df_comb2.empty:
                df_comb2 = df_comb.copy()

            # 抓出最終_總不平衡量最小的作為回傳結果
            comb_index1 = df_comb2["最終_總不平衡量"].idxmin()
            result1 = df_comb.loc[[comb_index1]]


            self.logging.info("Find second combination.")
            l_weight_fill = result1.loc[comb_index1, "平衡_L側配重"]
            f_weight_fill = result1.loc[comb_index1, "平衡_F側配重"]

            # 平衡配重的個位數去尾後的值
            l_quotient  = (l_weight_fill // 10) * 10
            f_quotient  = (f_weight_fill // 10) * 10

            # 產生可能的次要可行解組合，ex: [n, n+5, n+10]
            l_answers = [l_quotient, l_quotient + 5, l_quotient + 10] if (l_quotient >= 30) else []
            f_answers = [f_quotient, f_quotient + 5, f_quotient + 10] if (f_quotient >= 30) else []

            # 抓出符合的組合
            if (l_quotient >= 30) and (f_quotient >= 30): # 兩個都大於30
                df_comb3 = df_comb2.query(f"(平衡_L側配重 in {l_answers}) and (平衡_F側配重 in {f_answers})")
            elif (l_quotient >= 30) or (f_quotient >= 30): # 其中一個大於30 
                df_comb3 = df_comb2.query(f"((平衡_L側配重 in {l_answers}) and (平衡_F側配重 == {f_weight_fill})) or ((平衡_F側配重 in {f_answers}) and (平衡_L側配重 == {l_weight_fill}))")

            # 抓出最終_總不平衡量最小的作為回傳結果
            if ('df_comb3' in locals()) and (len(df_comb3) > 0):
                comb_index2 = df_comb3["最終_總不平衡量"].idxmin()
                result2 = df_comb.loc[[comb_index2]]
            else:
                result2 = pd.DataFrame()


            self.logging.info("Combine result.")
            results = []
            result_col = ["l_angle_pred", "l_weight_pred", "f_angle_pred", "f_weight_pred"]
            result1 = result1[['平衡_L側角度', '平衡_L側配重', '平衡_F側角度', '平衡_F側配重']].astype(str)
            result1.columns = result_col
            result1 = result1.to_dict(orient = "records")[0]
            results.append(result1)

            if len(result2) > 0:
                result2 = result2[['平衡_L側角度', '平衡_L側配重', '平衡_F側角度', '平衡_F側配重']].astype(str)
                result2.columns = result_col
                result2 = result2.to_dict(orient = "records")[0]
                results.append(result2)


            result = {
                "status":   "success",
                "work_id":  self.work_id,
                "op":       self.op,
                "model_id": self.model_id,
                "predict":  results
            }


        except:
            message = format_exc()
            result  = error(self.logging, message, self.model_id)


        finally:
            logging.info(f'Save output to {self.output_json}')
            with open(self.output_json, 'w') as file:
                json.dump(result, file, indent = 4)



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "predict.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "predict")
    
    logging.info("-"*200)
    # logging.info(f"root: {root}")
    

    input_ = get_input(sys.argv, logging)
    logging.info(f"input = {input_}")


    model = Model(root, input_, logging)
    model.predict()
            
    log.shutdown()