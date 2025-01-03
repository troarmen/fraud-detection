import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
data = pd.read_csv('creditcard_2023.csv', nrows=100000)

# Separate features and labels
X = data.drop('Class', axis=1)  # All features except the label
y = data['Class']  # Fraud label (1 = fraud, 0 = not fraud)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print("Accuracy:", accuracy) # Percentage of all predictions that were correct.
print("Precision:", precision) # Measures how many transactions flagged as fraud were actually fraud. High precision means fewer false positives.
print("Recall:", recall) # Measures how many actual fraud cases were correctly detected. High recall means fewer false negatives.


