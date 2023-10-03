import os, json, sys, pickle, warnings
import pandas as pd
from traceback import format_exc
from log_config import Log
from tools import *
from model import *

warnings.filterwarnings("ignore")



class Model():
    def __init__(self, root, input_, logging):
        self.logging  = logging
        self.model_id = input_["model_id"]


        # 取得train位置
        train_path = os.path.join(root, "data", "train")        
        os.makedirs(train_path, exist_ok = True)
        self.data_csv    = os.path.join(train_path, "train_data.csv")
        self.output_json = os.path.join(train_path, "output.json")


        # 取得model位置
        self.model_path = os.path.join(root, "data", "train", self.model_id)
        os.makedirs(self.model_path, exist_ok = True)
        
        self.model_detail = os.path.join(self.model_path, "model")
        os.makedirs(self.model_detail, exist_ok = True)



    def save_model(self, side, model):
        pickle.dump(model, open(os.path.join(self.model_detail, f"{side}_model.pkl"), "wb"))
    
    
    
    def train(self, df, aluminum_division, random_state, side):
        df = df[[f"初始_{side}側角度", f"初始_{side}側不平衡量"]]

        self.logging.info("- Feature engineering.")
        angle_init = np.linspace(0, 360, (aluminum_division + 1)).astype(int)
        df = calculate_angle_proportion(df, angle_init, aluminum_division, side)   
        df = calculate_weight(df, side)
        df1 = encoding(df, side)
        X_train, X_test, y_train, y_test = split_data(df1, random_state)

        self.logging.info("- Modeling.")
        models = modeling(X_train, y_train, random_state)
        scores, pred_trains, pred_tests = calculate_score(models, X_train, X_test, y_train, y_test, scoring = "weighted", cv_flag = 1)
        scores = scores.sort_values(("f1", "test"), ascending = False)
        best_model = scores.index[0]
        model = models[best_model]
        best_score = scores.loc[best_model, ("f1", "test")]

        self.logging.info(f'- Save {side} score and chart.')
        scores.to_csv(os.path.join(self.model_path, f"{side}_score.csv"))
        pred_plot(y_train, y_test, pred_trains, pred_tests, scores.loc[[best_model]], self.model_path, side)

        self.logging.info(f'- Save {side} model to {self.model_detail}\*.pkl')
        self.save_model(side, model)


        return best_score
    
    
    
    def run(self, aluminum_division = 12, random_state = 99):
        try:
            self.logging.info(f"Get data from {self.data_csv}")

            df_raw = pd.read_csv(self.data_csv)

            if df_raw.empty:
                raise NoDataFoundException
                    
            # df_all = df_raw.groupby("工號").first().reset_index(drop = True)
            df_all = df_raw.query("(初始_L側不平衡量 >= 4) & (初始_F側不平衡量 >= 4)").reset_index(drop = True)

            self.logging.info("Train L side data.")
            l_best_score = self.train(df_all, aluminum_division, random_state, side = "L")
            
            self.logging.info("Train F side data.")
            f_best_score = self.train(df_all, aluminum_division, random_state, side = "F")


            result = {
                "status":     "success",
                "model_id":   self.model_id,
                "accuracy":   min(l_best_score, f_best_score),
                "l_accuracy": l_best_score,
                "f_accuracy": f_best_score,
                }


        except (pd.errors.EmptyDataError, NoDataFoundException):
            message = "No data is available."
            result  = error(self.logging, message, self.model_id)
        

        except FileNotFoundError:
            message = "File not found."
            result  = error(self.logging, message, self.model_id)
        
        
        except:
            message = format_exc()
            result  = error(self.logging, message, self.model_id)


        finally:
            self.logging.info(f'Save output to {self.output_json}')
            with open(self.output_json, 'w') as file:
                json.dump(result, file, indent = 4)
    


class NoDataFoundException(Exception):
    pass



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "train.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "train")
    
    logging.info("-"*200)
    # logging.info(f"root: {root}")
    

    input_ = get_input(sys.argv, logging)
    logging.info(f"input = {input_}")


    model = Model(root, input_, logging)
    model.run()
            
    log.shutdown()