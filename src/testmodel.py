import pandas as pd
from joblib import load
from preprocess import preprocess_data

# Step 1: Load the test dataset
test_data = pd.read_csv('test.csv')

# Step 2: Preprocess the test data
X_test_new, _ = preprocess_data(test_data)  # Get only the features from the preprocessing

# Check the shape and columns of the preprocessed data
print("Shape of X_test_new:", X_test_new.shape)
print("Columns of X_test_new:", X_test_new.columns)

# Step 3: Load the trained model
model = load('models/final_model.pkl')

# Step 4: Make predictions on the preprocessed test dataset
predictions = model.predict(X_test_new)

# Step 5: Map predictions to cover type names (optional)
cover_type_mapping = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}
predicted_cover_types = [cover_type_mapping[pred] for pred in predictions]

# Step 6: Save the predictions to a CSV file
output = pd.DataFrame({'Id': test_data.index, 'Cover_Type': predicted_cover_types})
output.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
