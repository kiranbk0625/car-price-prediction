# 📦 Install required packages if not already done
# pip install kagglehub pandas scikit-learn matplotlib seaborn joblib

import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Download dataset from Kaggle using kagglehub
print("[INFO] Downloading dataset...")
path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
csv_path = f"{path}/car data.csv"
print("[INFO] Dataset downloaded to:", csv_path)

# Step 2: Load dataset
df = pd.read_csv(csv_path)
print("[INFO] First 5 rows:\n", df.head())

# Step 3: Feature Engineering
df['Car_Age'] = 2025 - df['Year']  # Assuming current year is 2025
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Step 4: Encode categorical variables
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le_seller.fit_transform(df['Seller_Type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

# Step 5: Split features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)

print("\n[RESULT] Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Step 9: Save model and encoders
joblib.dump(model, "car_price_model.pkl")
joblib.dump(le_fuel, "fuel_encoder.pkl")
joblib.dump(le_seller, "seller_encoder.pkl")
joblib.dump(le_trans, "trans_encoder.pkl")

print("\n[INFO] Model and encoders saved successfully.")
