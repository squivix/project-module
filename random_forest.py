import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler  # For undersampling

# Load the training and testing datasets
train_data = pd.read_csv('train_dataset.csv')  # Replace with your training data file
test_data = pd.read_csv('test_dataset.csv')    # Replace with your testing data file

# Split features and labels for training and testing sets
X_train = train_data.iloc[:, :-1]  # All columns except the last one are features
y_train = train_data.iloc[:, -1]   # The last column is the label
X_test = test_data.iloc[:, :-1]    # All columns except the last one are features
y_test = test_data.iloc[:, -1]     # The last column is the label

# Undersample the training data
rus = RandomUnderSampler(random_state=42)  # Random undersampling
X_train_resampled, y_train_resampled =X_train, y_train# rus.fit_resample(X_train, y_train)

# Train the Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient:", mcc)
