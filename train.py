
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

from mlrun.frameworks.sklearn import apply_mlrun
def train(context,random_i,tag):
    # Load the dataset
    iris = datasets.load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_i)
    
    # Create an instance of the model
    clf = svm.SVC()
    apply_mlrun(context=context,model=clf,model_name='model-test',tag=tag)
    # Train the model
    clf.fit(X_train, y_train)

