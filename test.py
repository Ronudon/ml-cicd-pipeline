import boto3
import pickle
import pandas as pd
from io import BytesIO

# Download model from S3
s3 = boto3.client("s3")
bucket_name = "ml-models-bucket"  # Replace with your S3 bucket name
model_key = "model/trained_model.pkl"

response = s3.get_object(Bucket=bucket_name, Key=model_key)
model_data = response["Body"].read()
model = pickle.loads(model_data)

# Test data (example)
test_data = pd.DataFrame({"size": [3000], "bedrooms": [3]})

# Make a prediction
predicted_price = model.predict(test_data)
print(f"🏡 Predicted house price: ${predicted_price[0]:,.2f}")
