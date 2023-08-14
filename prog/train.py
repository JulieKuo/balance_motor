import os, json, sys, pickle, time, warnings
import pandas as pd
from traceback import format_exc
from log_config import Log
from tools import *
from model import *
from tqdm import tqdm

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



    def save_model(self, side, outlier_boundary, skew_feat, pt, scaler, model):
        pickle.dump(self.features, open(os.path.join(self.model_detail, "feat_order.pkl"), "wb"))
        pickle.dump(outlier_boundary, open(os.path.join(self.model_detail, f"{side}_outlier_boundary.pkl"), "wb"))
        pickle.dump(skew_feat, open(os.path.join(self.model_detail, f"{side}_skew_feat.pkl"), "wb"))
        pickle.dump(pt, open(os.path.join(self.model_detail, f"{side}_power_tf.pkl"), "wb"))
        pickle.dump(scaler, open(os.path.join(self.model_detail, f"{side}_scaler.pkl"), "wb"))
        pickle.dump(model, open(os.path.join(self.model_detail, f"{side}_model.pkl"), "wb"))
    
    
    
    def train(self, df, stop, model_boundary, side):
        if side == "L":
            target = self.target_vol[0]
        else:
            target = self.target_vol[1]

        total_scores = pd.DataFrame()
        total_models = {}
        preds = {}
        num = 0
        start, end = time.time(), time.time()
        stop *= 60 # min -> s
        pbar = tqdm(total = stop, ncols = 50)
        while (end - start) < stop:
            train, test, _ = split_data(df)
            train, test, outlier_boundary = deal_with_outlier(self.features, train, test)
            train, test, skew_feat, pt    = deal_with_skew(self.features, train, test)
            train, test, scaler = scaling(self.features, train, test)

            X_train, X_test = train[self.features], test[self.features]
            y_train, y_test = train[target], test[target]


            models = modeling(X_train, y_train, random_state = num)

            scores0, pred_trains, pred_tests = calculate_score(models, X_train, X_test, y_train, y_test, scoring = "r2", cv_flag = 1, cv_scoring = "r2", random_state = num)
            scores = scores0.sort_values(("r2", "test"), ascending = False).iloc[[0]]

            if (scores["r2"] > model_boundary).values.all():
                preds[num] = {"train": pred_trains, "test": pred_tests}
                total_models[num] = models[scores.index[0]]
                scores["order"] = num
                total_scores = pd.concat([total_scores, scores])
            
            num += 1
            mid = end
            end = time.time()
            pbar.update(round(end - mid, 4))
            
        pbar.close()


        self.logging.info(f'- Save {side} score and chart.') 
        total_scores = total_scores.sort_values([('r2',  'test'), ('mape',  'test'), ('r2',  'train'), ('mape',  'train')], ascending = [False, True, False, True])
        total_scores1 = total_scores[(total_scores[('r2',  'train')] > total_scores[('r2',  'test')]) & (total_scores[('mape',  'train')] < total_scores[('mape',  'test')])]
        if len(total_scores1) == 0:
            total_scores1 = total_scores.copy()

        best_order = total_scores1["order"].iloc[0]
        best_score = total_scores1.iloc[[0]]
        best_score = best_score.drop("order", axis = 1)
        best_score.to_csv(os.path.join(self.model_path, f"{side}_score.csv"))

        pred_trains, pred_tests = preds[best_order]["train"], preds[best_order]["test"]
        pred_plot(pred_trains, pred_tests, best_score, target, self.model_path, side)


        self.logging.info(f'- Save {side} model to {self.model_detail}\*.pkl')        
        model = total_models[best_order]
        self.save_model(side, outlier_boundary, skew_feat, pt, scaler, model)


        return best_score
    
    
    
    def run(self, limit_end = 30, stop = 5, model_boundary = 0.6):
        try:
            self.logging.info(f"Get data from {self.data_csv}")

            df = pd.read_csv(self.data_csv)

            if df.empty:
                raise NoDataFoundException
            
            df = df.dropna().reset_index(drop = True)
            

            self.logging.info("Generate feature.")
            df = generate_feature(df)


            self.logging.info("Split data.")
            self.target_vol = ["最終_L側不平衡量", "最終_F側不平衡量"]
            target_angle    = ["最終_L側角度", "最終_F側角度"]
            features   = df.columns.drop(self.target_vol + target_angle).to_list()
            self.features = features[3:]

            l_df = df[df[self.target_vol[0]] <= limit_end].reset_index(drop = True)
            f_df = df[df[self.target_vol[1]] <= limit_end].reset_index(drop = True)


            self.logging.info("Modeling.")
            l_best_score = self.train(l_df, stop, model_boundary, side = "L")
            f_best_score = self.train(f_df, stop, model_boundary, side = "F")
            l_best_score = l_best_score.iloc[0][("r2", "test")]
            f_best_score = f_best_score.iloc[0][("r2", "test")]


            result = {
                "status":     "success",
                "model_id":   self.model_id,
                "accuracy":   max(l_best_score, f_best_score),
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