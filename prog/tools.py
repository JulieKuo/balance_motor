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

        df.loc[i, ["angle_proportion", "angle_0", "angle_target"]] = [angle_proportion, angle_0, angle_target]
    
    return df


def solve_integer_linear_program(y):
    # 建立整數線性規劃求解器
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # 創建整數變數
    x4 = solver.IntVar(0, solver.infinity(), 'x4')
    x5 = solver.IntVar(0, solver.infinity(), 'x5')
    x6 = solver.IntVar(0, solver.infinity(), 'x6')
    x7 = solver.IntVar(0, solver.infinity(), 'x7')
    x10 = solver.IntVar(0, solver.infinity(), 'x10')
    x20 = solver.IntVar(0, solver.infinity(), 'x20')
    x30 = solver.IntVar(0, solver.infinity(), 'x30')

    # 新增約束條件
    solver.Add((4 * x4) + (5 * x5) + (6 * x6) + (7 * x7) + (10 * x10) + (20 * x20) + (30 * x30) == y)
    solver.Minimize(x4 + x5 + x6 + x7 + x10 + x20 + x30)

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


def calculate_weight(df, side):
    for i in range(len(df)):
        weight = df.loc[i, f"初始_{side}側不平衡量"]
        for proportion in [[3, 7], [5, 5], [10, 0]]:
            weight1 = round(weight * (proportion[0]/10))
            weight2 = round(weight * (proportion[1]/10))
            
            if weight >= 4:
                solution1, weight1 = solve_combination(weight1)
                solution2, weight2 = solve_combination(weight2)
            else:                
                solution1, weight1 = {}, 0
                solution2, weight2 = {}, 0

            
            if proportion == [3, 7]:
                df.loc[i, [f"{proportion[0]}", f"{proportion[1]}", f"{str(proportion[0]) + str(proportion[1])}_delta", f"{proportion[0]}_solution", f"{proportion[1]}_solution"]] = [
                    weight1, weight2, abs(weight - (weight1 + weight2)), solution1, solution2]
            else:
                df.loc[i, [f"{proportion[0]}", f"{proportion[0]}_delta", f"{proportion[0]}_solution"]] = [
                    weight1, abs(weight - (weight1 + weight2)), solution1]
                    
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