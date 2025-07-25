# 🚗 Car Price Prediction App

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/ML%20Model-RandomForestRegressor-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Complete-brightgreen)

A simple yet powerful machine learning web app that predicts the **resale value** of a used car based on inputs like fuel type, transmission, KMs driven, and more.

---

## 📸 Screenshot

![App Screenshot](screenshots/app-ui.png)

---

## 📊 Features

- Interactive UI using **Streamlit**
- Real-world dataset from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- Trained on `RandomForestRegressor`
- Model saved and reused via `joblib`
- Categorical variables encoded with `LabelEncoder`
- Predicted output: **Estimated price in ₹ lakhs**

---

## 🚀 Demo

```bash
streamlit run app.py
