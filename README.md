# Making Predictions on the Iris Dataset with Argparse

## Install

This project requires Python 3.6 and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install [Anaconda](https://www.anaconda.com/distribution/) distribution of Python, which 
already has the above packages, and make sure to select Python 3.x Installer.

## Run

In a terminal or command window, navigate to the notebook directory and run one of the following commands:

`iris.ipynb`

This will open the Jupyter Notebook software and project file in your browser.

You can also open the files of python, I use pycharm to bring forward sintaks from `iris.ipynb` to `predic.py` and `train.py`, in those files I use argparse Python standard library, so we can call the parsing argument in the command-line.

## General Documentation

To running the `predic.py` :

```html
$ python predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2
iris_setosa 
```
Here is what is happening : 

- Running the script with python and call the file file (predic.py) that will we use.
- Call the arguments (--sepal_length, --sepal_width, --petal_length, --petal_width) and put content of features, than will show the species of Iris.

To running the `train.py` :

```html
$ python train.py --path=dataset/iris.data 
 sepal_length  sepal_width  petal_length  petal_width         species
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa

```
Running the script with call the argumen of --path and choose directory the place of file iris.data saved.
```html
$ python train.py --path=dataset/iris.data --feature_column=sepal_width,sepal_length,petal_width,petal_length 
 sepal_width  sepal_length  petal_width  petal_length
```

```html
$ python train.py --path=dataset/iris.data --feature_column=sepal_width,sepal_length,petal_width,petal_length --target_column=species 
0      0
1      0
2      0
3      0
4      0
```

```html
$ python train.py --path=dataset/iris.data --feature_column=sepal_width,sepal_length,petal_width,petal_length --target_column=species --algorithm=svm 
[[47  0  0]
 [ 0 38  4]
 [ 0  2 44]]
Accuracy Detection: 0.9703703703703703
```

```html
$ python train.py --path=dataset/iris.data --feature_column=sepal_width,sepal_length,petal_width,petal_length \--target_column=species --algorithm=svm  --algorithm=svm --save_to=weight/desy.pkl
```

```html
$ python train.py --path=dataset/iris.data --feature_column=sepal_width,sepal_length,petal_width,petal_length \--target_column=species --algorithm=svm  --algorithm=svm --save_to=/home/abbiyanaila/Desktop/fandi.pkl
```

Here is what is happening :

- 



## Data

The dataset used in this project is included as `iris.data` in dataset directory. This dataset is a freely available on the [UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/iris). This dataset has the following attributes:

#### Features

`features`: sepal_length , sepal_width, petal_length, petal_width

`target`: species

After the training in Jupyter Notebook done, save the file with pkl file extension and save to weight directory. 

## Outcome

I basically used 2 types of classifier to predict the outcome of the Iris types.

`SVM` : The accuracy of SVM classifier turned out to be 0.9703703703703703
.

`DecisionTree` : The accuracy of DecisionTree classifier turned out to be 1.0.

