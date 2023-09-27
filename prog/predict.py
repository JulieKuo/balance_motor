import os, sys, json, pickle
import pandas as pd
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
        self.l_weight_ori = float(input_["l_weight_ori"])
        self.f_angle_ori  = int(input_["f_angle_ori"])
        self.f_weight_ori = float(input_["f_weight_ori"])
        self.material     = input_["material"]



    def load_model(self, side):
        model = pickle.load(open(os.path.join(self.model_detail, f"{side}_model.pkl"), "rb"))


        return model
    
    
    def predict(self, aluminum_division, side, angle_ori, weight_ori):
        self.logging.info(f'- Load model from {self.model_detail}\*.pkl')
        model = self.load_model(side)

        self.logging.info("- Feature engineering.")
        df = pd.DataFrame([[angle_ori, weight_ori]], columns = [f"初始_{side}側角度", f"初始_{side}側不平衡量"])
        angle_init = np.linspace(0, 360, (aluminum_division + 1)).astype(int)
        df = calculate_angle_proportion(df, angle_init, aluminum_division, side)        
        df = calculate_weight(df, side)
        df1 = encoding(df, side)
        X, _ = split_data(df1, predict = True)

        self.logging.info("- Predicting.")
        y_pred = model.predict(X)[0]

        return y_pred, df
        
    
    
    def run(self, aluminum_division = 12):
        try:
            self.logging.info("Predict L side data.")
            y_pred_l, df_l = self.predict(aluminum_division, side = "L", angle_ori = self.l_angle_ori, weight_ori = self.l_weight_ori)
            

            self.logging.info("Predict F side data.")
            y_pred_f, df_f = self.predict(aluminum_division, side = "F", angle_ori = self.f_angle_ori, weight_ori = self.f_weight_ori)


            self.logging.info("Create result.")
            # 轉換type
            astype_feats = ['3', '7', '5', '10']
            df_l[astype_feats] = df_l[astype_feats].astype(int).astype(str)
            df_f[astype_feats] = df_f[astype_feats].astype(int).astype(str)
            l_angle_ori = str(self.l_angle_ori)
            f_angle_ori = str(self.f_angle_ori)

            # 最佳補償值
            predict = [
                    {
                        "l_angle_pred":  l_angle_ori,
                        "l_weight_pred": df_l.loc[0, "10"],
                        "f_angle_pred":  f_angle_ori,
                        "f_weight_pred": df_f.loc[0, "10"]
                    }
            ]
            predict.append({})

            # 組合
            comb = {
                df_l.loc[0, "10"]: df_l.loc[0, "10_solution"],
                df_f.loc[0, "10"]: df_f.loc[0, "10_solution"]
            }

            # F側補償值組合
            if y_pred_l == 0:
                predict[1].update({
                    "l_angle_pred":  l_angle_ori,
                    "l_weight_pred": [df_l.loc[0, "3"], df_l.loc[0, "7"]],
                })
                comb.update({df_l.loc[0, "3"]: df_l.loc[0, "3_solution"]})
                comb.update({df_l.loc[0, "7"]: df_l.loc[0, "7_solution"]})

            elif y_pred_l == 1:
                predict[1].update({
                    "l_angle_pred":  l_angle_ori,
                    "l_weight_pred": [df_l.loc[0, "5"], df_l.loc[0, "5"]],
                })
                comb.update({df_l.loc[0, "5"]: df_l.loc[0, "5_solution"]})

            # L側補償值組合
            if y_pred_f == 0: 
                predict[1].update({
                    "f_angle_pred":  f_angle_ori,
                    "f_weight_pred": [df_f.loc[0, "3"], df_f.loc[0, "7"]],
                })
                comb.update({df_f.loc[0, "3"]: df_f.loc[0, "3_solution"]})
                comb.update({df_f.loc[0, "7"]: df_f.loc[0, "7_solution"]})

            elif y_pred_f == 1:
                predict[1].update({
                    "f_angle_pred":  f_angle_ori,
                    "f_weight_pred": [df_f.loc[0, "5"], df_f.loc[0, "5"]],
                })
                comb.update({df_f.loc[0, "5"]: df_f.loc[0, "5_solution"]})            


            result = {
                "status":   "success",
                "work_id":  self.work_id,
                "op":       self.op,
                "model_id": self.model_id,
                "predict":  predict,
                "comb":     comb
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
    model.run()
            
    log.shutdown()