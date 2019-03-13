import argparse
import pandas as pd
import numpy as np


""" command-line parsing in the Python standard library """
parser = argparse.ArgumentParser(description='Prediction of Iris')
parser.add_argument('--sepal_length', type=float, help='sepal length of Iris')
parser.add_argument('--sepal_width', type=float, help='sepal width of iris')
parser.add_argument('--petal_length', type=float, help='petal length of iris')
parser.add_argument('--petal_width', type=float, help='petal width of iris')
parser.add_argument('--load', default=None, help='you cal load file from the folder')
args = parser.parse_args()


""" fungtion of predict_iris and four parameter"""
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """ declaration of numpy array from parameter """
    test = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    """ load file .pkl from directory """
    if args.load == None:
        clf = pd.read_pickle('weight/iris.sklearn.pkl')
        pred = clf.predict(test)
    """ or load from another directory that will we choose """
    else:
        clf = args.load
        pred = clf.predict(test)
    """ return from the frist array """
    return pred[0]


""" a built-in variable which evaluates to the name of the current module. """
if __name__ == '__main__':
    """ make labels for prediction results """
    label = ['iris_setosa', 'iris_versicolor', 'iris_virginica']
    """ predict of iris if we fill in the number of features """
    out = predict_iris(args.sepal_length, args.sepal_width,
                       args.petal_length, args.petal_width)
    """ print the result of predict """
    print(f'Predict Detection is: {label[out]}')