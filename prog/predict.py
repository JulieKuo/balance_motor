import os, sys, json, pickle
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



    def get_input(self, input_):
        self.work_id      = input_["work_id"]
        self.op           = input_["op"]
        self.model_id     = input_["model_id"]
        self.speed        = int(input_["speed"])
        self.l_angle_ori  = int(input_["l_angle_ori"])
        self.l_weight_ori = round(float(input_["l_weight_ori"]))
        self.f_angle_ori  = int(input_["f_angle_ori"])
        self.f_weight_ori = round(float(input_["f_weight_ori"]))



    def load_model(self, side):
        features         = pickle.load(open(os.path.join(self.model_detail, f"feat_order.pkl"), "rb"))
        outlier_boundary = pickle.load(open(os.path.join(self.model_detail, f"{side}_outlier_boundary.pkl"), "rb"))
        skew_feat        = pickle.load(open(os.path.join(self.model_detail, f"{side}_skew_feat.pkl"), "rb"))
        pt               = pickle.load(open(os.path.join(self.model_detail, f"{side}_power_tf.pkl"), "rb"))
        scaler           = pickle.load(open(os.path.join(self.model_detail, f"{side}_scaler.pkl"), "rb"))
        model            = pickle.load(open(os.path.join(self.model_detail, f"{side}_model.pkl"), "rb"))


        return features, outlier_boundary, skew_feat, pt, scaler, model



    def predict(self, gap = 40, min_change = 20):
        try:
            self.logging.info(f'Load model from {self.model_detail}\*.pkl')
            features, l_outlier_boundary, l_skew_feat, l_pt, l_scaler, l_model = self.load_model(side = "L")
            features, f_outlier_boundary, f_skew_feat, f_pt, f_scaler, f_model = self.load_model(side = "F")
            
            
            self.logging.info("Generate candidate values.")
            l_weight_change, f_weight_change = [], []
            for i in range(gap+1):
                l_weight_change.extend([max(min_change, self.l_weight_ori + i), max(min_change, self.l_weight_ori - i)])
                f_weight_change.extend([max(min_change, self.f_weight_ori + i), max(min_change, self.f_weight_ori - i)])
            l_weight_change = list(set(l_weight_change))
            f_weight_change = list(set(f_weight_change))

            l_angle_change0, f_angle_change0 = [], []
            for i in range(0, gap + 1, 10):
                l_angle_change0.extend([self.l_angle_ori + i, self.l_angle_ori - i])
                f_angle_change0.extend([self.f_angle_ori + i, self.f_angle_ori - i])
            l_angle_change0 = list(set(l_angle_change0))
            f_angle_change0 = list(set(f_angle_change0))

            l_angle_change = []
            for angle in l_angle_change0:
                if angle > 360:
                    l_angle_change.append(angle - 360)
                elif angle < 0:
                    l_angle_change.append(360 + angle)
                else:
                    l_angle_change.append(angle)

            f_angle_change = []
            for angle in f_angle_change0:
                if angle > 360:
                    f_angle_change.append(angle - 360)
                elif angle < 0:
                    f_angle_change.append(360 + angle)
                else:
                    f_angle_change.append(angle)
            
            
            self.logging.info("Generate candidate combinations.")
            all_combinations = []
            for a in l_angle_change:
                for b in l_weight_change:
                    for c in f_angle_change:
                        for d in f_weight_change:
                            x = [self.speed, self.l_angle_ori, self.l_weight_ori, self.f_angle_ori, self.f_weight_ori, a, b, c, d]
                            all_combinations.append(x)
            
            df_comb = pd.DataFrame(
                all_combinations, 
                columns = ['平衡轉速', '初始_L側角度', '初始_L側不平衡量', '初始_F側角度', '初始_F側不平衡量', '平衡_L側角度', '平衡_L側配重', '平衡_F側角度', '平衡_F側配重']
                )
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
            df_comb["總不平衡量"] = df_comb.eval("(abs(最終_L側不平衡量) + abs(最終_F側不平衡量))")

            # 先嚴格篩選，在寬鬆篩選
            df_comb1 = df_comb.query("((平衡_L側配重 > 0) and (平衡_F側配重 > 0)) and ((最終_L側不平衡量 >= 0.1) and (最終_F側不平衡量 >= 0.1))")
            if len(df_comb1) == 0:
                df_comb1 = df_comb.query("(最終_L側不平衡量 >= 0.1) and (最終_F側不平衡量 >= 0.1)")
            if len(df_comb1) == 0:
                df_comb1 = df_comb.copy()
                
            comb_index = df_comb1["總不平衡量"].idxmin()
            result = df_comb.loc[[comb_index]]

            result = result[['平衡_L側角度', '平衡_L側配重', '平衡_F側角度', '平衡_F側配重']]
            result = result.astype(str)
            result.columns = ["l_angle_pred", "l_weight_pred", "f_angle_pred", "f_weight_pred"]
            result = result.to_dict(orient = "records")[0]


            result = {
                "status":   "success",
                "work_id":  self.work_id,
                "op":       self.op,
                "model_id": self.model_id,
                "predict":  result
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