import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset (CSV or Excel)
# Change the file path based on your dataset location
file_path = r".\DATA\processed\training_data.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# If you're using an Excel file, use this instead:
# df = pd.read_excel('your_dataset.xlsx')

# Step 2: Preprocessing

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# If there are missing values, you can either drop them or fill them
# For this example, let's assume there are no missing values after preprocessing

# Separate features (X) and target (y)
X = df.drop('appendicitis', axis=1)  # Assuming 'appendicitis' is the target column
y = df['appendicitis']

# If you have categorical columns, you can encode them using one-hot encoding
# For example:
# X = pd.get_dummies(X)  # If you have categorical variables to encode

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (important for some algorithms, but Random Forest is less sensitive)
# For Random Forest, scaling is not as critical, but we'll scale features here for consistency.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest
rf_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optionally: Save the trained model using pickle
import pickle
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)
