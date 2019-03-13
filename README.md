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

