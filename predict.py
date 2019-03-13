import argparse
import pandas as pd
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Prediction of Iris')
parser.add_argument('--sepal_length', type=float, help='sepal length of Iris')
parser.add_argument('--sepal_width', type=float, help='sepal width of iris')
parser.add_argument('--petal_length', type=float, help='petal length of iris')
parser.add_argument('--petal_width', type=float, help='petal width of iris')
args = parser.parse_args()


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    test = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    clf = pd.read_pickle('weight/iris.sklearn.pkl')
    pred = clf.predict(test)
    return pred[0]

if __name__ == '__main__':
    label = ['iris_setosa', 'iris_versicolor', 'iris_virginica']
    out = predict_iris(args.sepal_length, args.sepal_width,
                       args.petal_length, args.petal_width)
    print(label[out])