from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)

X_train, y_train, X_test, y_test = train_test_split(X, y)