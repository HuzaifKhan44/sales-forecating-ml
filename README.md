# sales-forecating-ml
Retail Sales Forecasting using XGBoost with Real-World Data
# 🧠 Sales Forecasting using Machine Learning

This project presents a machine learning pipeline to forecast daily sales in retail stores using real-world time series data.

---

## 📌 Project Overview

Retailers rely heavily on accurate sales forecasts for inventory management, staffing, and marketing. In this project, we develop a forecasting model using historical sales, promotional data, holiday information, and engineered features like lagged values and rolling averages.

- **Goal**: Predict daily sales per store-family combination
- **Data**: Inspired by the Favorita Store Sales Dataset (Kaggle)
- **Approach**: Gradient boosting with XGBoost + feature engineering
- **Output**: Trained model, EDA visuals, insights, and business recommendations

---

## 🧪 Technologies Used

- **Language**: Python
- **Libraries**: Pandas, NumPy, XGBoost, Scikit-learn, Matplotlib, Seaborn
- **Model**: XGBoost Regressor
- **Tools**: Jupyter Notebook, PowerPoint (for presentation), joblib (for model persistence)

---

## 📁 Project Structure

SalesForecastingProject/
├── train_processed.csv # Cleaned training dataset
├── test_processed.csv # Cleaned test dataset
├── xgboost_sales_model.pkl # Trained model (saved with joblib)
├── sales_forecasting_model.py # Model training & prediction script
├── Sales_Forecasting_Insights_Presentation.pptx # Key insights presentation
└── Sales_Forecasting_Report.pdf # (Optional) 20-page detailed report


---

## 📊 Key Features

- ✅ Time-based feature engineering (lag, rolling mean, day of week)
- ✅ Promotion & holiday handling
- ✅ Actual vs Predicted plots
- ✅ Error distribution & feature importance visualizations
- ✅ Confidence interval forecasting
- ✅ Strategic business recommendations

---

## 📈 Sample Results

| Metric          | Value     |
|-----------------|-----------|
| RMSE            | ~274.2    |
| MAE             | ~198.7    |
| R² Score        | ~0.74     |

> ✅ Model captures weekly seasonality, promotional effects, and general sales trends well.

---

## 📉 Visual Outputs

- `actual_vs_predicted.png`: Overlay of true and predicted sales  
- `error_distribution.png`: Histogram of residuals  
- `feature_importance.png`: Top predictive features from XGBoost  
- `forecast_with_confidence.png`: Future forecast with uncertainty range  

*(Available in presentation slides)*

---

## 🚀 How to Run

```bash
# Step 1: Install requirements
pip install xgboost pandas scikit-learn joblib

# Step 2: Run the training script
python sales_forecasting_model.py

🧠 Business Applications
🔄 Inventory and supply chain optimization

🎯 Promotion scheduling around demand spikes

📅 Workforce planning based on predicted sales volume

📊 BI integration for automated forecasting dashboards

🔮 Future Enhancements
Deploy a Streamlit web app for real-time predictions

Use LSTM for multi-step temporal forecasting

Incorporate external economic/transactional indicators

Implement live model monitoring and feedback loop


🙋‍♂️ Author
Huzaif Ulla Khan
Data Science Enthusiast
Email:khuzaif319@gmail.com

⭐ If you find this project useful, feel free to star the repo or fork it to build your own version!
