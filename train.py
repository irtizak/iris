from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import numpy as np

def run():
    # Set random seed
    rseed = 786

    # Load data
    X, y = load_iris(return_X_y=True)

    # Create train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rseed)

    # Declare model
    clf = RandomForestClassifier(random_state=rseed)

    # Fit model
    clf.fit(X_train, y_train)

    # Make predictions
    y_preds = clf.predict(X_test)

    # Calculate metric
    model_error = round(np.sqrt(mean_squared_error(y_preds, y_test)), 2)

    # Print metric
    print(model_error)

if __name__=='__main__':
    run()