# Customer Churn Prediction with XGBoost & Streamlit

A supervised machine learning web application that predicts customer churn probability using behavioral and service-related features. The project compares multiple ML algorithms and deploys the best-performing model through an interactive **Streamlit** dashboard for real-time insights.

***

## ✨ Features

- Multi-model training and comparison (Logistic Regression, SVM, Random Forest, XGBoost, etc.).
- Hyperparameter tuning using GridSearchCV for optimal performance.
- Leakage-free training pipeline with careful feature selection.
- Feature importance analysis to understand key churn drivers.
- Real-time churn probability scoring for individual customers.
- Risk classification into Low, Moderate, and High churn risk.
- Deployment-ready Streamlit dashboard for business stakeholders.

***

## 🚀 Live Demo

Explore the deployed **Churn Prediction Dashboard** on Hugging Face Spaces:

➡️ [Churn Prediction Dashboard – Hugging Face Space](https://huggingface.co/spaces/aditii27/Churn-Prediction-Dashboard)

The app lets you input customer details, view predicted churn probability, and see the assigned risk category in real time.

***

## 🚀 Tech Stack

| Component        | Technology                              |
|------------------|------------------------------------------|
| Machine Learning | XGBoost, Logistic Regression, SVM, Random Forest |
| Data Processing  | Pandas, NumPy                           |
| Model Tuning     | GridSearchCV                            |
| Scaling          | StandardScaler                          |
| Web Framework    | Streamlit                               |
| Language         | Python 3.x                              |
| Deployment       | Hugging Face Spaces                     |

***

## 📁 Project Structure

```bash
Churn_prediction/
├── Data/
│   └── processed_data.csv
│
├── Models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── Notebooks/
│   ├── Churn_EDA_Preprocessing.ipynb
│   └── Model_Training.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

***

## ⚙️ Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/aditii27/Churn_prediction.git
   cd Churn_prediction
   ```

2. **(Optional) Create and activate a virtual environment**
   ```bash
   python -m venv venv
   ```

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux / macOS:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

   Then open `http://localhost:8501` in your browser.

***

## 🧠 How It Works

- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical variables for stable model training.
- **Leakage Removal**: Excludes features such as Churn Score, CLTV, Latitude, Longitude, and Zip Code to avoid data leakage and ensure fair evaluation.
- **Model Training**: Trains multiple supervised ML models (Logistic Regression, SVM, Random Forest, XGBoost, etc.).
- **Model Selection**: Selects XGBoost as the final model based on highest Recall and ROC-AUC, prioritizing churn detection.
- **Deployment**: Integrates the best model into a Streamlit dashboard for real-time prediction and risk visualization.

***

## 📊 Model Performance

Final selected model: **XGBoost**.

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~0.75  |
| Recall    | ~0.80  |
| F1 Score  | ~0.63  |
| ROC-AUC   | ~0.85  |

The model prioritizes **Recall** to effectively identify customers who are likely to churn, which is crucial for retention strategies.

***

## 🔍 Top Feature Importance

Most influential churn predictors include:

- Tenure Months  
- Total Charges  
- Contract Type  
- Monthly Charges  
- Dependents  
- Internet Service (Fiber optic)  
- Payment Method  

These features align well with real-world drivers of customer churn in subscription-based services.

***

## 💻 Sample Results

Add your dashboard screenshots under an `images/` folder and reference them here, for example:

![Hish Risk Output](images/output1.jpg)
![Low Risk Output](images/output2.png)

***

## 🎯 Risk Classification

The application categorizes customers into churn risk bands based on predicted probability:

- 🔴 **High Risk** → Probability > 0.65  
- 🟠 **Moderate Risk** → 0.40 – 0.65  
- 🟢 **Low Risk** → Probability < 0.40  

This segmentation helps teams design targeted retention and outreach campaigns.

***

## 🤝 Contributing

Contributions are welcome. To contribute:

- Fork the repository.  
- Create a new branch:  
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Commit your changes:  
  ```bash
  git commit -m "Add feature"
  ```
- Push to your branch:  
  ```bash
  git push origin feature/your-feature-name
  ```
- Open a Pull Request.

***

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project with proper attribution.

***

## 👩‍🎓 Author

**Aditi Soni**  
- GitHub: [github.com/aditii27](https://github.com/aditii27)

If you found this project helpful, consider starring the repository ⭐.

***
