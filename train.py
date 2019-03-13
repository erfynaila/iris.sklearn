import argparse
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pickle
import datetime
from sklearn.model_selection import cross_val_predict

parser = argparse.ArgumentParser(description="Training Iris")
parser.add_argument('path', help='load file with format csv')
parser.add_argument('features_X', type=str, help='features of iris')
parser.add_argument('target', type=int, help='target of iris')
parser.add_argument('X', type=str, help='featuter')
parser.add_argument('y', type=int, help='target')
parser.add_argument('train_size', type=float, help='break into two parameter X and y')
parser.add_argument('random_state', type=int, help='break into two parameter X and y')
parser.add_argument('standard_scaler', help='standardize features by removing the mean&scaling')
parser.add_argument('fit_transform', help='fit to data, then transform it')
parser.add_argument('training', help='process of training')
parser.add_argument('cross_validation', help='predict the training of target')
parser.add_argument('confusion_matrix', help='confusion matrix from predic')
parser.add_argument('score_accuration', help='score accuration of training')
parser.add_argument('datetime', help='for knowing the time if the file will to save')
parser.add_argument('save_file', help='save file .pkl')
parser.add_argument('load_file', help='load file .pkl')

df = pd.read_csv("dataset/iris.data")

col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df.loc[:, col]

species_to_num = {'Iris-setosa': 0,
                  'Iris-versicolor': 1,
                  'Iris-virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)

clf = svm.SVC(kernel='linear', C=1.0, verbose=True)
clf.fit(X_std_train, y_train)

y_train_pred = cross_val_predict(clf, X_std_train, y_train, cv=3)

confusion_matrix(y_train, y_train_pred)

clf.score(X_std_train, y_train)

dstr= datetime.datetime.now().strftime("%y%m%d%H%M%S")
pkl_filename = "iris.sklearn_"+dstr+".pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


