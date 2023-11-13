import base64, json
from traceback import format_exc
from ortools.linear_solver import pywraplp
from sklearn.model_selection import train_test_split
import numpy as np



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


def calculate_angle_proportion(df, angle_init, aluminum_division, side):
    for i in range(len(df)):
        angle = df.loc[i, f"初始_{side}側角度"]
        closest_index = np.argsort(np.abs(angle_init - angle))[0] # angle_init中與angle_ori最接近的角度的index
        near_angle = angle_init[closest_index] # 最接近的角度
        angle_proportion = abs(angle - near_angle) / (360 / aluminum_division) # 平衡角與最近可填補角度的距離佔每格的比例
        angle_0 = 0

        if angle_proportion == 0:
            angle_0 = 1
            angle_target = "10"
        elif angle_proportion <= 0.3:
            angle_target = "37"
        else:
            angle_target = "55"
        
        prop_37 = "37" if near_angle > angle else "73"

        df.loc[i, ["angle_proportion", "angle_0", "angle_target", "prop_37"]] = [angle_proportion, angle_0, angle_target, prop_37]
    
    return df


def solve_integer_linear_program(y, count_limit = 4):
    # 建立整數線性規劃求解器
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # 創建整數變數
    x4 = solver.IntVar(0, solver.infinity(), 'x4')
    x5 = solver.IntVar(0, solver.infinity(), 'x5')
    x6 = solver.IntVar(0, solver.infinity(), 'x6')
    x7 = solver.IntVar(0, solver.infinity(), 'x7')
    x16 = solver.IntVar(0, solver.infinity(), 'x16')
    x20 = solver.IntVar(0, solver.infinity(), 'x20')

    # 新增約束條件
    solver.Add((4 * x4) + (5 * x5) + (6 * x6) + (7 * x7) + (16 * x16) + (20 * x20) == y)
    solver.Add(x4 + x5 + x6 + x7 + x16 + x20 <= count_limit)
    solver.Minimize(x4 + x5 + x6 + x7 + x16 + x20)

    # 求解問題
    status = solver.Solve()

    # 回傳結果
    solution = {}
    if status == pywraplp.Solver.OPTIMAL:
        for x in solver.variables():
            if x.solution_value() != 0:
                solution[x.name().strip("x")] = str(round(x.solution_value()))
        return True, solution
    else:
        return False, solution


def solve_combination(weight):    
    result, solution = solve_integer_linear_program(y = weight)
    
    # 如果沒有解，依序加減1，找解
    weight1, weight2 = weight, weight
    while not result:
        weight1 = max(weight1 - 1, 0)
        weight2 += 1
        # print(weight, weight1, weight2)

        result1, solution1 = solve_integer_linear_program(y = weight1) # 往上加1，查看有無解
        result2, solution2 = solve_integer_linear_program(y = weight2) # 往上減1，查看有無解

        # 若有解則跳出迴圈
        if result1:
            weight = weight1
            result, solution = result1, solution1
        elif result2:
            weight = weight2
            result, solution = result2, solution2
    
    return solution, int(weight)


def calculate_weight(df, side, weight_limit = 56):
    weight_col = f"初始_{side}側不平衡量"
    # 按權重拆分重量
    for proportion in [[3, 7], [5, 5], [10, 0]]:
        df[f"{proportion[0]}_raw"] = df[weight_col] * (proportion[0]/10)
        if proportion == [3, 7]:
            df[f"{proportion[1]}_raw"] = df[weight_col] * (proportion[1]/10)

    # 按weight_limit拆分重量，每個角不可超過weight_limit
    df[["3_full", "7_full", "5_full", "10_full"]] = (df[["3_raw", "7_raw", "5_raw", "10_raw"]] // weight_limit).astype(int)
    df[["3", "7", "5", "10"]] = df[["3_raw", "7_raw", "5_raw", "10_raw"]] % weight_limit
    df[["3_amt", "7_amt", "5_amt", "10_amt"]] = 1


    # 修正10的補償值
    minus_index = df[(df["10_full"] % 2 == 0) & (df["10_full"] != 0)].index
    df.loc[minus_index, "10_full"] -= 1
    df.loc[minus_index, "10"] += weight_limit
    split_index = df[df["10_raw"] > weight_limit].index
    df.loc[split_index, "10"] /= 2
    df.loc[split_index, "10_amt"] += 1


    # 計算最適組合並更新建議重量
    for col, values in df[["3", "7", "5", "10"]].items():
        solutions = []
        new_weights = []
        for weight in values:
            solution, weight = solve_combination(round(weight))
            solutions.append(solution)        
            new_weights.append(weight)
        
        df[col] = new_weights
        df[f"{col}_solution"] = solutions

    # 計算誤差
    for cols in [["3", "7"], ["5"], ["10"]]:
        delta = ((df[f"{cols[0]}_full"] * weight_limit) + (df[f"{cols[0]}"] * df[f"{cols[0]}_amt"])) - df[f"{cols[0]}_raw"]
        if cols == ["3", "7"]:
            delta += ((df[f"{cols[1]}_full"] * weight_limit) + (df[f"{cols[1]}"] * df[f"{cols[1]}_amt"])) - df[f"{cols[1]}_raw"]
        
        new_col = "".join(cols)
        df[f"{new_col}_delta"] = delta
                    
    return df
    

def encoding(df, side):
    df1 = df.copy()
    df1["angle_target"] = df1["angle_target"].map({"37": 0, "55": 1, "10": 2})
    df1[['3', '7', '5', '10']] = df1[['3', '7', '5', '10']].values / df1[f"初始_{side}側不平衡量"].values.reshape(len(df1), 1)
    df1 = df1.fillna(0)

    return df1


def split_data(df, random_state = 99, predict = False):
    X = df[['angle_proportion', 'angle_0', '3', '7', '37_delta', '5', '5_delta', '10', '10_delta']]
    y = df['angle_target']
    if predict:
        return X, y


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state, stratify = y)
    
    return X_train, X_test, y_train, y_test



def fill_solution(prop, ans, predict, weight_pred, weight_limit, side, comb_50 = {"16": "1", "20": "2"}):
    for i, col in enumerate(prop):
        # 重量超過weight_limit時，有幾個角填補weight_limit
        if ans[f"{col}_full"] != 0:
            predict[f"{side}_comb"][weight_pred[i]] = {
                str(weight_limit):{
                    "count": str(round(ans[f"{col}_full"])),
                    "comb": comb_50
                }
            }
        else:
            predict[f"{side}_comb"][weight_pred[i]] = {}

        # 最後一個角填補的值，不超過weight_limit
        if ans[col] != 0:
            predict[f"{side}_comb"][weight_pred[i]][str(ans[col])] = {
                "count": str(ans[f"{col}_amt"]),
                "comb": ans[f"{col}_solution"]
            }
            
    return predict



def chart_comb(best, predicts, side = "l"):
    weight0 = str(predicts[best][f"{side}_weight_pred"][0])
    comb0 = predicts[best][f"{side}_comb"][weight0]

    if best != "1:0":
        weight1 = str(predicts[best][f"{side}_weight_pred"][1])
        comb1 = predicts[best][f"{side}_comb"][weight1]

        left = []
        for key, vlaue in comb0.items():
            left.extend([key] * int(vlaue["count"]))

        right = []
        for key, vlaue in comb1.items():
            right.extend([key] * int(vlaue["count"]))

        left  = [int(i) for i in left]
        right = [int(i) for i in right]
        left  = sorted(left, reverse = True)
        right = sorted(right, reverse = True)
        left  = [str(i) for i in left]
        right = [str(i) for i in right]

    else:
        comb0_keys = [int(i) for i in comb0.keys()]
        comb0_sort = sorted(comb0_keys, reverse = True)

        left = []
        right = []
        for comb in comb0_sort:
            count = int(comb0[str(comb)]["count"])
            while count >= 2:
                left.append(comb)
                right.insert(0, comb)
                count -= 2

            if count != 0:
                left.append(comb)

    return left, right