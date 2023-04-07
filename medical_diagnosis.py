from sklearn.metrics import accuracy_score
from mindsdb_native import Predictor
from mindsdb_native.libs.constants.mindsdb import *
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('synthetic_medical_data.csv')

X = data.drop(columns=['Diagnosis'])  # Features
y = data['Diagnosis']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a MindsDB instance
mdb = Predictor(name='medical_diagnosis_model')

# Train the model
mdb.learn(from_data=pd.concat([X_train, y_train], axis=1),
          to_predict='Diagnosis', use_gpu=False, stop_training_in_x_seconds=60)

# Get predictions for the test_data
predictions = mdb.predict(when_data=X_test)

predicted_values = [prediction['Diagnosis'] for prediction in predictions]
actual_values = y_test.tolist()

accuracy = accuracy_score(actual_values, predicted_values)
print("Model Accuracy:", accuracy)

# Print the shapes of the subsets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Print the first few rows of the training set
print("\nX_train:")
print(X_train.head())
print("\ny_train:")
print(y_train.head())

# Print the first few rows of the testing set
print("\nX_test:")
print(X_test.head())
print("\ny_test:")
print(y_test.head())
