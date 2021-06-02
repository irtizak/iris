# This file is to test ideas
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()

print(data['feature_names'])

print(data['data'])

df = pd.DataFrame(columns = data['feature_names'], data = data['data'])

print(df.describe())

print(data['target_names'])

df = load_iris(as_frame=True)

print(df.keys())