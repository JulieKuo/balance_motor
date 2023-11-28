import os, json, sqlalchemy, sys
import pandas as pd
from traceback import format_exc
from log_config import Log
from tool import *


class Export():
    def __init__(self, root, input_):
        # 取得output位置
        export_path = os.path.join(root, "data", "export")
        os.makedirs(export_path, exist_ok = True)
        self.output_xlsx = os.path.join(export_path, "data.xlsx")
        self.output_json = os.path.join(export_path, "output.json")

        # 取得config
        config_path = os.path.join(root, "prog", "config.json")
        with open(config_path) as f:
            self.config = json.load(f)
        

        # 資料區間
        self.start_time = input_["start_time"]
        self.end_time = input_["end_time"]
    


    def data_excel(self):
        try:
            logging.info(f'Read data from database.')
            con_info = f'mysql+pymysql://{self.config["user"]}:{self.config["password"]}@{self.config["host"]}:{self.config["port"]}/{self.config["database"]}'
            conn = sqlalchemy.create_engine(con_info)

            query = f'SELECT * FROM {self.config["table"]} WHERE  (modify_by != "admin") AND ((create_time >= "{self.start_time}") AND (create_time <= "{self.end_time}"))'
            df = pd.read_sql(query, conn)

            if df.empty:
                raise NoDataFoundException
            

            logging.info('Data processing.')


            # 欄位名稱
            col_ch = [
                ('',                     '日期'),
                ('動平衡作業提升紀錄表',   '工號(範例)'),
                ('動平衡作業提升紀錄表',   '序號'),
                ('動平衡作業提升紀錄表',   '平衡轉速\nrpm'),
                ('初始不平衡值角度和重量', 'L側角度\nLA'),
                ('初始不平衡值角度和重量', 'L側\n不平衡量(g)'),
                ('初始不平衡值角度和重量', 'F側角度\nFA'),
                ('初始不平衡值角度和重量', 'F側\n不平衡量(g)'),
                ('平衡塊安裝角度值和重量', 'L側角度\n=LA'),
                ('平衡塊安裝角度值和重量', 'L側配重(g)'),
                ('平衡塊安裝角度值和重量', 'F側角度\n=FA'),
                ('平衡塊安裝角度值和重量', 'F側配重(g)'),
                ]
            col_ch = pd.MultiIndex.from_tuples(col_ch)
            col_mid = col_ch.get_level_values(1).to_list()
            col_mid[1] = "工號"

            # 抓出需匯出的欄位
            col_sort = ['create_time', 'work_id', 'op', 'speed', 'l_angle_ori', 'l_weight_ori', 'f_angle_ori', 'f_weight_ori', 'l_angle_act', 'l_weight_act', 'f_angle_act', 'f_weight_act']
            df = df[col_sort]

            # 刪除重複出現的樣本
            df = df.drop_duplicates(subset = ["work_id", "op"], keep = "last")

            # 資料庫中的create_time一開始建時有缺失值，須補齊
            df = df.sort_values(["work_id", "op"])
            df["create_time"] = df["create_time"].fillna(method = "bfill").fillna(method = "ffill")

            # 轉換輸出值         
            df["create_time"] = df["create_time"].dt.date # time to date
            df["op"] = "第" + df["op"].astype(str) + "次"
            df["op"] = df["op"].replace("第1次", "初始不平衡值")

            # 轉換輸出格式
            df_first = df[df["op"] == "初始不平衡值"].sort_values(["create_time"]) # 按日期排序工單

            new_df = pd.DataFrame()
            g = df.groupby("work_id")
            for group in df_first["work_id"]:
                df1 = g.get_group(group)
                df1.iloc[1:, :2] = None

                # 新增中間欄位名
                if not new_df.empty:
                    col_mid[0] = df1.iloc[0, 0]
                    df1.iloc[0, 0] = None
                    new_df = pd.concat([new_df, pd.DataFrame([col_mid], columns = new_df.columns)], ignore_index = True)
                    
                new_df = pd.concat([new_df, df1], ignore_index = False)


            logging.info(f'Save data to {self.output_xlsx}')
            new_df.columns = col_ch
            new_df.to_excel(self.output_xlsx, index=True)


            result = {
                "status": "success"
                }        
        
        
        except (pd.errors.EmptyDataError, NoDataFoundException):
            message = "該區段查無資料"
            result  = error(logging, message)
                

        except:
            message = format_exc()
            result  = error(logging, message)


        finally:
            logging.info(f'Save output to {self.output_json}')
            with open(self.output_json, 'w', encoding = 'utf-8') as file:
                json.dump(result, file, indent = 4, ensure_ascii = False)
            
            log.shutdown()


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
    logging = log.set_log(filepath = os.path.join(log_path, "export.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "export")
    
    logging.info("-"*100)
    # logging.info(f"root: {root}")


    input_ = get_input(sys.argv, logging)
    logging.info(f"input = {input_}")
    

    export = Export(root, input_)
    export.data_excel()