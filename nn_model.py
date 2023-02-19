from itertools import cycle
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def plotTraining(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    #ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()

    plt.ylabel('merror')
    plt.title('XGBoost Multiclass Classification Error')
    plt.show()

def optimizeHyperParams(x_data,y_data):
    data_dmatrix = xgb.DMatrix(data=x_data,label=y_data)
    for i in range(1,15):
        x = i + 1
        params = {"objective": "multi:softmax", 'learning_rate': 0.36, 'max_depth': 5, 'alpha': 1}
        print(repr(x))
        print(paramTest(params,data_dmatrix))


def paramTest(params, dMat):
    cv_results = xgb.cv(dtrain=dMat, params=params, metrics="rmse", as_pandas=True, seed=1234)
    return cv_results.tail(1)


def runModel(X_train,X_test,y_train,y_test):
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        learning_rate='0.36',
        max_depth=5,
        alpha=1,
        eval_metric='mlogloss',
        early_stopping_rounds=15
    )

    model.fit(X_train,y_train, verbose=0, eval_set=[(X_test,y_test)])

    #results = model.evals_result()
    model.save_model('basicModel.json')
    #plotTraining(model)


def train(data):
    # Sort between data used to classify and data to predict
    target_Label = ["Hand_Signal"]
    train_Labels = trainLabels()
    x_data = data[train_Labels]
    y_data = data[target_Label]

    X_train, x_nonTrain, y_train,y_nonTrain = train_test_split(x_data.values,y_data.values,test_size = 0.2, random_state = 1234)
    x_test,x_validate, y_test,y_validate = train_test_split(x_nonTrain,y_nonTrain,test_size=0.5,random_state=1234)

    # properly encode labels
    le = LabelEncoder()
    le.fit([10,20,30,40])
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    runModel(X_train,x_test,y_train,y_test)
    #optimizeHyperParams(x_validate,y_validate)
    classify(x_validate,y_validate)


def trainLabels():
    # Create a list of the form ['x0','y0','x1','y1',...]
    col_name = []
    # Creates a list of the form [0,0,1,1,...]
    for i in range(21):
        col_name.append(i)
        col_name.append(i)
    col_name = [str(x) for x in col_name]
    coord_labels = ["x", "y"]
    # Create list of tuples
    col_name = zip(cycle(coord_labels), col_name)
    labels = []
    # Create strings
    for blah in col_name:
        labels.append(''.join(blah[0] + blah[1]))
    return labels


# Loads all for csv's and combines into a single DataFrame
def loadData():
    thumbsUp = pd.read_csv('thumbsup.csv')
    thumbsDown = pd.read_csv('thumbsdown.csv')
    openPalm= pd.read_csv('testOpenPalm.csv')
    closedFist = pd.read_csv('testClosedFist.csv')
    intermediaryOne = pd.concat([thumbsUp,thumbsDown])
    intermediaryTwo = pd.concat([openPalm,closedFist])
    data = pd.concat([intermediaryOne, intermediaryTwo])
    return data

def classify(data,other_data):
    hand_classifer = xgb.XGBClassifier()
    hand_classifer.load_model('basicModel.json')
    print(data[0].shape)
    print(hand_classifer.get_booster().feature_names)
    for i in range(10):
        x = hand_classifer.predict(data[i].reshape(1,42))
        print(x)
        print(other_data[i])
        print(data[i].reshape(1,42))

def main():
    data = loadData()
    train(data)


if __name__ == "__main__":
    main()