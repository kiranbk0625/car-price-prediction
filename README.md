🚗 Car Price Prediction System (Regression)
🔍 Project Overview:
Build a machine learning model that predicts the price of a used car based on features like brand, fuel type, transmission, mileage, year, etc.

It’s a regression problem, not classification — so the goal is to predict a numeric value (price) instead of a category.

📁 Dataset:
Use this Kaggle dataset:
🚗 Used Cars Dataset – India (Kaggle)

Use kagglehub to download:

python
Copy
Edit
import kagglehub
path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
print("Dataset downloaded at:", path)
📌 Features You’ll Work With:
Feature	Description
Car_Name	Name of the car
Year	Year the car was bought
Selling_Price	Target: price to predict (in lakhs)
Present_Price	Price when car was new
Kms_Driven	Kilometers driven
Fuel_Type	Petrol / Diesel / CNG
Seller_Type	Dealer / Individual
Transmission	Manual / Automatic
Owner	0/1/2/3 – Number of previous owners

🧠 What You'll Learn:
Feature Engineering (e.g., age of car = current year - purchase year)

Label Encoding for categorical features

Train/Test split + Evaluation

Regression models like LinearRegression, RandomForestRegressor, or XGBoost

R² Score, MAE, RMSE

📦 Libraries Needed:
bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
