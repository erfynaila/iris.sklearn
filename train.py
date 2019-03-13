import argparse
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pickle
import datetime
from sklearn.model_selection import cross_val_predict
from sklearn import tree


""" command-line parsing in the Python standard library """
parser = argparse.ArgumentParser(description="Training Iris")
parser.add_argument('--path', type=str, help='load file with format csv')
parser.add_argument('--feature_column', type=str, help='feature from iris dataset')
parser.add_argument('--target_column', type=str, help='target from iris dataset')
parser.add_argument('--train_size', default=0.9, type=float, help='give train size')
parser.add_argument('--random_state', default=0, type=int, help='give random state')
parser.add_argument('--algorithm', type=str, help='you can choose algorithm that you want to use')
parser.add_argument('--save_to', default=None, help='you file will save in folder weight')
args = parser.parse_args()


""" call file of csv in path, so we can call the path from command-line """
df = pd.read_csv(args.path)
""" determine the features column for training """
col = args.feature_column.split(',')
X = df.loc[:, col]


""" change the sample targets to number and determine targets column for training """
species_to_num = {'Iris-setosa': 0,
                  'Iris-versicolor': 1,
                  'Iris-virginica': 2}
df['tmp'] = df[args.target_column].map(species_to_num)
y = df['tmp']


""" split (breaks into two parameters namely X and y) """
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=args.train_size,
                                                    random_state=args.random_state)

""" variabel with standarscaler function, fit to variable then transform it"""
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)


""" choose algorithm that will use for training """
if args.algorithm == 'svm':
    clf = svm.SVC(kernel='linear', C=1.0 )
    clf.fit(X_std_train, y_train)
elif args.algorithm == 'tree':
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_std_train, y_train)


""" variable with cross_val_predic function for variale and features of training """
y_train_pred_svm = cross_val_predict(clf, X_std_train, y_train, cv=3)
""" generate guessed data """
conf_mat = confusion_matrix(y_train, y_train_pred_svm)
""" to see the accuration score training """
score = clf.score(X_std_train, y_train)


""" save to file in the current working directory with datetime for code of file """
if args.save_to == None:
    dstr = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    pkl_filename = "weight/test.sklearn_" + dstr + ".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
else:
    pkl_filename = args.save_to
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)




