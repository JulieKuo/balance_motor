import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score, fbeta_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']



def modeling(X_train, y_train, random_state = 99):
    models = {        
        "Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state = random_state),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        # print(name + " trained.")
    
    return models



def calculate_score(models, X_train, X_test, y_train, y_test, scoring = "weighted", cv_flag = 1):
    round_ = 2
    average = scoring
    col = [("accuracy", "train"), ("accuracy", "test"), ("precision", "train"), ("precision", "test"), ("recall", "train"), ("recall", "test"), ("f1", "train"), ("f1", "test")]

    cv_flag = 1
    if cv_flag:
        col.append((("accuracy", "cv")))
    col = pd.MultiIndex.from_tuples(col)

    pred_trains = pd.DataFrame(y_train)
    pred_tests = pd.DataFrame(y_test)
    scores = pd.DataFrame()

    for name, model in models.items():
        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)
        pred_trains[f"{name}_train"] = pred_train
        pred_tests[f"{name}_test"] = pred_test

        acc_train  = accuracy_score(y_train, pred_train).round(round_)
        acc_test   = accuracy_score(y_test, pred_test).round(round_)

        precision_train  = precision_score(y_train, pred_train, average = average).round(round_)
        precision_test   = precision_score(y_test, pred_test, average = average).round(round_)

        recall_train  = recall_score(y_train, pred_train, average = average).round(round_)
        recall_test   = recall_score(y_test, pred_test, average = average).round(round_)

        f1_train = f1_score(y_train, pred_train, average = average).round(round_)
        f1_test = f1_score(y_test, pred_test, average = average).round(round_)

        score = [acc_train, acc_test, precision_train, precision_test, recall_train, recall_test, f1_train, f1_test]

        if cv_flag:
            cv_scores = cross_val_score(model, X_train, y_train, cv = 3, scoring = 'accuracy')
            cv_score = cv_scores.mean().round(round_)
            score.append(cv_score)
        
        scores.loc[name, col] = score
    
    return scores, pred_trains, pred_tests



def pred_plot(y_train, y_test, pred_trains, pred_tests, scores, model_path, side, flag = False):
    for name in scores.index:
        fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_train, pred_trains[f"{name}_train"]), display_labels = (set(y_train) | set(pred_trains[f"{name}_train"])))
        disp.plot(cmap = plt.cm.Blues, ax = ax[0])
        ax[0].set_title(f"{side} - train - {name}")

        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, pred_tests[f"{name}_test"]), display_labels = (set(y_test) | set(pred_tests[f"{name}_test"])))
        disp.plot(cmap = plt.cm.Blues, ax = ax[1])
        ax[1].set_title(f"{side} - test - {name}")
        
        if flag:
            plt.show()

        print(f"Classification Report of {side} Test:\n{classification_report(y_test, pred_tests[f'{name}_test'])}")
        print("-"*100)
    
    fig.savefig(f"{model_path}/{side}_cunfusion_matrix.png")