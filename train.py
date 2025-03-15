import pandas as pd
import boto3
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset (Replace with your dataset)
data = pd.DataFrame({
    "size": [4174, 4507, 1860, 2294, 2130, 2095, 4772],
    "bedrooms": [2, 2, 4, 2, 2, 4, 4],
    "price": [416437, 469848, 199339, 234262, 226013, 214914, 490572]
})

# Split into training and testing sets
X = data[["size", "bedrooms"]]
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Upload model to S3
s3 = boto3.client("s3")
bucket_name = "ml-models-bucket"  # Replace with your S3 bucket name
s3.upload_file("trained_model.pkl", bucket_name, "model/trained_model.pkl")

print("✅ Model trained and uploaded to S3 successfully!")
