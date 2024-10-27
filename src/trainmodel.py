import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from joblib import dump
from preprocess import preprocess_data, load_data

# Load and preprocess data
data = load_data('train.csv')
X, y = preprocess_data(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save the model
# Use an absolute path
model_dir = 'forestcoverpredmlproj/models'

# Create directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Path to save the model
model_path = os.path.join(model_dir, 'final_model.pkl')

# Save the model
dump(model, model_path)

# Step 6: Model Evaluation
predictions = model.predict(X_test)
print("Evaluation Report:")
print(classification_report(y_test, predictions))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.show()

# Step 7: Model Tuning (Hyperparameter Tuning)
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

rf_random = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3)
rf_random.fit(X_train, y_train)

# Evaluate the tuned model
tuned_predictions = rf_random.best_estimator_.predict(X_test)
print("Tuned Model Evaluation Report:")
print(classification_report(y_test, tuned_predictions))
