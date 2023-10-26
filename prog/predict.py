import os, sys, json, pickle
import pandas as pd
from traceback import format_exc
from log_config import Log
from tools import *



class Model():
    def __init__(self, root, input_, logging):
        self.logging = logging

        
        # 角度調整選項
        self.adjustments = {
            1: -15, 
            2: -7.5,
            3: 0, 
            4: 7.5, 
            5: 15
        }


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
    


    def adjust_angle(self, angle, adjustment):
        angle += adjustment
        if angle < 0:
            angle += 360

        elif angle >= 360:
            angle -= 360

        return round(angle)



    def get_input(self, input_):
        self.work_id      = input_["work_id"]
        self.op           = input_["op"]
        self.model_id     = input_["model_id"]
        self.speed        = int(input_["speed"])
        self.adjust       = int(input_["adjust"])
        self.l_angle_ori = int(input_["l_angle_ori"])
        self.l_angle_ori1  = self.adjust_angle(self.l_angle_ori, self.adjustments[self.adjust]) # 原點角度調整
        print(f"l: {self.l_angle_ori} -> {self.l_angle_ori1}")
        self.l_weight_ori = float(input_["l_weight_ori"])
        self.f_angle_ori = int(input_["f_angle_ori"])
        self.f_angle_ori1  = self.adjust_angle(self.f_angle_ori, self.adjustments[self.adjust]) # 原點角度調整
        print(f"f: {self.f_angle_ori} -> {self.f_angle_ori1}")
        self.f_weight_ori = float(input_["f_weight_ori"])
        self.material     = input_["material"]



    def load_model(self, side):
        model = pickle.load(open(os.path.join(self.model_detail, f"{side}_model.pkl"), "rb"))

        return model
    
    
    def predict(self, aluminum_division, side, angle_ori, weight_ori, weight_limit, l_extra = 6.4, f_extra = 8):
        self.logging.info(f'- Load model from {self.model_detail}\*.pkl')
        model = self.load_model(side)

        self.logging.info("- Feature engineering.")
        df = pd.DataFrame([[angle_ori, weight_ori]], columns = [f"初始_{side}側角度", f"初始_{side}側不平衡量"])

        # 超過60補上特定值
        extra_amount = l_extra if side == "L" else f_extra
        if df.loc[0, f"初始_{side}側不平衡量"] > weight_limit:
            df.loc[0, f"初始_{side}側不平衡量"] += extra_amount

        angle_init = np.linspace(0, 360, (aluminum_division + 1)).astype(int)
        df = calculate_angle_proportion(df, angle_init, aluminum_division, side)
        df = calculate_weight(df, side)
        df1 = encoding(df, side)
        X, _ = split_data(df1, predict = True)

        self.logging.info("- Predicting.")
        y_pred = model.predict(X)[0]

        return y_pred, df
        
    
    
    def run(self, aluminum_division = 12, weight_limit = 56):
        try:
            self.logging.info("Predict L side data.")
            y_pred_l, df_l = self.predict(aluminum_division, side = "L", angle_ori = self.l_angle_ori1, weight_ori = self.l_weight_ori, weight_limit = weight_limit)
            

            self.logging.info("Predict F side data.")
            y_pred_f, df_f = self.predict(aluminum_division, side = "F", angle_ori = self.f_angle_ori1, weight_ori = self.f_weight_ori, weight_limit = weight_limit)


            self.logging.info("Create result.")
            l_ans = df_l.iloc[0]
            f_ans = df_f.iloc[0]

            proportion = {    
                "1:0": ["10"],
                "5:5": ["5", "5"],
                "3:7": ["3", "7"]
            }

            predicts = {}
            for key, value in proportion.items():
                l_weight_pred = [str(round((l_ans[f"{value[0]}_full"] * weight_limit) + l_ans[value[0]]))]
                f_weight_pred = [str(round((f_ans[f"{value[0]}_full"] * weight_limit) + f_ans[value[0]]))]

                if value != ["10"]: # "1:0"只有一個解
                    l_weight_pred += [str(round((l_ans[f"{value[1]}_full"] * weight_limit) + l_ans[value[1]]))]
                    f_weight_pred += [str(round((f_ans[f"{value[1]}_full"] * weight_limit) + f_ans[value[1]]))]

                # 初始化該比例下的predict
                predict = {
                    "l_angle_pred":  str(self.l_angle_ori),
                    "l_weight_pred": l_weight_pred,
                    "l_comb": {},
                    "f_angle_pred":  str(self.f_angle_ori),
                    "f_weight_pred": f_weight_pred,
                    "f_comb": {},
                }

                # 填入組合
                fill_solution(value, l_ans, predict, l_weight_pred, weight_limit, side = "l")
                fill_solution(value, f_ans, predict, f_weight_pred, weight_limit, side = "f")

                predicts[key] = predict


            # 最佳建議值
            y_map = {0: "3:7", 1: "5:5", 2: "1:0"}
            best = {
                "l_side": y_map[y_pred_l],
                "f_side": y_map[y_pred_f]
            }
            


            result = {
                "status":   "success",
                "work_id":  self.work_id,
                "op":       self.op,
                "model_id": self.model_id,
                "predict":  predicts,
                "best":     best
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