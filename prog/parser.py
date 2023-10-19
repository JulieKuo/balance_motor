import os, json, re, shutil
import pandas as pd
from traceback import format_exc
from log_config import Log


class Parser():
    def __init__(self, root):
        # 取得input和output位置
        upload_path = os.path.join(root, "data", "upload")
        files = os.listdir(upload_path) # 取得所有檔案和資料夾
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(upload_path, f))) # 依照更新日期排序，抓出最新的檔案
        self.input_csv = os.path.join(upload_path, latest_file)

        # 獲取clean資料夾位置並清空
        self.clean_path = os.path.join(root, "data", "clean")
        if os.path.exists(self.clean_path):
            shutil.rmtree(self.clean_path)   
        os.makedirs(self.clean_path, exist_ok = True)
        # self.output_csv = os.path.join(clean_path, "data.csv")
        self.output_json = os.path.join(self.clean_path, "output.json")
    
    def data_clean(self, local = True):
        try:
            logging.info(f'Read data from {self.input_csv}')
            df = pd.read_excel(self.input_csv, header = [0, 1])
            

            logging.info('Parsing.')


            # 修改column名
            df_col = []
            for name1, name2 in df.columns:
                name1 += "_"
                if ("Unnamed" in name1) or ("動平衡作業提升紀錄表" in name1) or ("備註" in name2):
                    name1 = ""
                name = name1 + name2
                name = re.sub(pattern = r"不平衡值角度和重量|塊安裝角度值和重量|\n|=|LA|FA|\(g\)|rpm", repl = "", string = name)
                name = name.replace("工號(範例)", "範例")
                df_col.append(name)
            df.columns = df_col


            # 補齊日期
            df["日期"] = df["日期"].fillna(method = "ffill")


            # 刪除row中重複出現的column名
            drop_id = df.query("(序號 == '序號')").index
            df = df.drop(drop_id, axis = 0).reset_index(drop = True)


            # 切割工號及範例，並補齊工號
            df.insert(loc = 1, column = "工號", value = None)
            id_dict = df.query("序號 == '初始不平衡值'")["範例"].to_dict()
            for index, id_ in id_dict.items():
                df.loc[index, "工號"] = id_
                df.loc[index, "範例"] = "初始"

            df["工號"] = df["工號"].fillna(method = "ffill")
            init_dict = {value: key for key, value in id_dict.items()}


            # 序號轉為數字
            df = df[~df.isna().all(axis = 1)].reset_index(drop = True)
            df["序號"] = [1 if "初始" in i else int(i.replace("第", "").replace("次", "")) for i in df["序號"]]


            # 刪除特徵
            drop_col = set(df.columns) & set(["Unnamed: 0_level_1"])
            df = df.drop(drop_col, axis = 1)


            # 補充平衡角度的資料
            df["平衡_L側角度"] = df["平衡_L側角度"].fillna(df["初始_L側角度"])
            df["平衡_F側角度"] = df["平衡_F側角度"].fillna(df["初始_F側角度"])


            # 角度文字轉數字
            transform_l = df["平衡_L側角度"].apply(lambda X: "度" in str(X)) # 抓出包含"度"的row
            transform_l = transform_l[transform_l].index.tolist()

            transform_f = df["平衡_F側角度"].apply(lambda X: "度" in str(X))
            transform_f = transform_f[transform_f].index.tolist()

            for i in transform_l:
                add = re.search(r'(\d+)[度]', df.loc[i, "平衡_L側角度"]).group(1) # 抓出"度"之前的數值
                df.loc[i, "平衡_L側角度"] = df.loc[i, "初始_L側角度"] + int(add) # 初始值加上該數值

            for i in transform_f:
                add = re.search(r'(\d+)[度]', df.loc[i, "平衡_F側角度"]).group(1)
                df.loc[i, "平衡_F側角度"] = df.loc[i, "初始_F側角度"] + int(add)


            # 新增最終值
            df["最終_L側角度"] = df["初始_L側角度"].shift(-1)
            df["最終_L側不平衡量"] = df["初始_L側不平衡量"].shift(-1)
            df["最終_F側角度"] = df["初始_F側角度"].shift(-1)
            df["最終_F側不平衡量"] = df["初始_F側不平衡量"].shift(-1)


            # 獲取達標數據的index (只抓取平衡量小於1之前的資料)
            reach_index = df.query("(初始_L側不平衡量 <= 1) and (初始_F側不平衡量 <= 1)").index
            reach_dict = df.loc[reach_index, "工號"].drop_duplicates(keep = "first").to_dict()
            reach_dict = {value: key for key, value in reach_dict.items()}


            # 不使用加扇後的資料
            df[["範例", "備註"]] = df[["範例", "備註"]].fillna("None")
            stop_df = df[df["範例"].str.contains("扇")]["工號"]
            stop_dict = stop_df.drop_duplicates(keep = "first").to_dict()
            for index, work_id in stop_dict.items():
                reach_dict[work_id] = min(reach_dict[work_id], index)


            # 獲取初始數據的index
            init_index = df.query("序號 == 1").index
            init_dict = df.loc[init_index, "工號"].to_dict()
            init_dict = {value: key for key, value in init_dict.items() if value in reach_dict.keys()}


            # 抓出要保留的資料
            keep_index = []
            for key in reach_dict.keys():
                keep_index.extend(range(init_dict[key], reach_dict[key]))
            df = df.iloc[keep_index].reset_index(drop = True)


            # 刪除連續變數中不為數值的資料
            check_num_col = ['平衡轉速', '初始_L側角度', '初始_L側不平衡量', '初始_F側角度', '初始_F側不平衡量', '平衡_L側角度', '平衡_F側角度']
            df.loc[:, check_num_col] = df.loc[:, check_num_col].apply(pd.to_numeric, errors = 'coerce')
            df = df[~df[check_num_col].isnull().any(axis = 1)]


            # 獲取材料為鋁的資料
            aluminum_flag = df["範例"].str.contains("鋁")
            aluminum_work_id = df["工號"][aluminum_flag]
            df = df.query("工號 in @aluminum_work_id").reset_index(drop = True)


            # 刪除剪枝和不補償後的資料
            df1 = pd.DataFrame()
            g = df.groupby("工號")
            for group in df["工號"].unique():
                df_group = g.get_group(group).reset_index(drop = True)
                stop_flag1 = df_group["範例"].str.contains("鉚合") # 剪枝tag
                stop_flag2 = df_group["備註"].str.contains("不補償") # 不補償tag
                stop_flag = (stop_flag1 | stop_flag2)
                if sum(stop_flag) != 0:        
                    stop_index = df_group[stop_flag].index[0]
                    df_group = df_group.iloc[:stop_index] # 只保留剪枝和不補償前的資料
                df1 = pd.concat([df1, df_group], ignore_index = True)
                
            df = df1.drop(["範例", "備註"], axis = 1)


            logging.info(f'Save data to {self.clean_path}\*.csv')

            # 逐日期存一個檔
            g = df.groupby("日期")
            for group in df["日期"].unique():
                df1 = g.get_group(group)
                date = str(group)[:10].replace("-", "")
                df1.to_csv(os.path.join(self.clean_path, f"{date}.csv"), index = False)
            
            if local:
                train_path = os.path.join(root, "data", "train")        
                os.makedirs(train_path, exist_ok = True)
                output_csv = os.path.join(train_path, "train_data.csv")
                logging.info(f'Save data to {output_csv}')
                df.to_csv(output_csv, index = False)

            result = {
                "status": "success",
                "reason": "",
                "data_counts": len(df),
                "job_counts": df["工號"].nunique(),
                "date_counts": df["日期"].nunique()
                }
                
        except:
            logging.error(format_exc())
            result = {
                "status": "fail",
                "reason": format_exc(),
                }

        finally:
            logging.info(f'Save output to {self.output_json}')
            with open(self.output_json, 'w') as file:
                json.dump(result, file, indent = 4)
            
            log.shutdown()



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "parser.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "parser")
    
    logging.info("-"*100)
    # logging.info(f"root: {root}")
    

    parser = Parser(root)
    parser.data_clean(local = True)