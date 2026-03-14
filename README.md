
# 🏭 AI Manufacturing Intelligence

AI Manufacturing Intelligence is a **machine learning–powered dashboard** that predicts pharmaceutical tablet quality based on manufacturing process parameters.
The system uses an **XGBoost model with a Streamlit interface** to help optimize production, improve quality control, and estimate energy consumption.

## 🚀 Features

**📊 Quality Prediction Dashboard**

* Predicts tablet quality metrics:

  * Hardness
  * Friability
  * Dissolution
  * Uniformity
  * Disintegration
* Displays results using visual indicators.

**🏆 Golden Batch Optimization**

* Suggests optimal compression force.
* Helps achieve the best production batch quality.

**🧠 Explainable AI**

* Uses **SHAP** to explain how each parameter affects predictions.

**⚡ Energy & Carbon Analytics**

* Estimates energy usage and CO₂ emissions.
* Compares baseline vs current production.

## 🧠 Model

The system uses an **XGBoost regression model** trained to predict multiple pharmaceutical quality metrics from manufacturing parameters.

**Input Parameters**

* Granulation Time
* Binder Amount
* Drying Temperature
* Drying Time
* Compression Force
* Machine Speed
* Lubricant Concentration
* Moisture Content

The trained model is stored as `xgboost_quality_model.pkl`. 

## 🛠️ Tech Stack

* Python
* Streamlit
* XGBoost
* Scikit-learn
* Pandas
* NumPy
* SHAP
* Matplotlib

Dependencies are listed in `requirements.txt`. 

## 📂 Project Structure

```
AI-Manufacturing-Intelligence
│
├── app.py
├── xgboost_quality_model.pkl
├── requirements.txt
└── README.md
```

## 👩‍💻 Author

**Bodapati Bhavani**
AI & Data Science Student

