# Customer Churn Prediction Project 📊

This project is a complete data science workflow aimed at predicting customer churn using a provided dataset. The primary goal is to build a reliable machine learning model that can identify customers who are likely to stop using a service, enabling proactive retention strategies.

The project progresses from initial data cleaning and exploratory analysis to building, comparing, and interpreting multiple machine learning models. The final result is a highly accurate **XGBoost** model with an accuracy of over 93%.

## 🚀 Key Insights & Findings

Our analysis revealed several key factors that strongly influence customer churn:

  * **Loyalty Program Engagement is Critical:** The most significant predictor of churn is a customer's engagement with the loyalty program. Features like **`points_in_wallet`** and **`membership_category`** are the most important factors.
  * **Behavioral Metrics Matter:** User activity, such as **`avg_transaction_value`**, **`avg_time_spent`**, and how recently they logged in, are also powerful predictors of churn.
  * **Individual Prediction Insights:** Using SHAP, we can explain precisely *why* a specific customer is flagged as a churn risk by breaking down the prediction based on their personal data (e.g., low wallet points, negative feedback).

-----

## 📂 Dataset

The project uses the `churn.csv` dataset, which contains 36,992 records and 25 columns detailing customer demographics, transaction history, membership status, and feedback.

**Target Variable:** `churn_risk_score` (1 for churn, 0 for no churn).

-----

## 🛠️ Project Workflow

The project follows a standard data science methodology:

1.  **Data Cleaning & Preprocessing:**

      * Handled inconsistent data (e.g., `'?'` characters).
      * Corrected data types for numerical and date columns.
      * Dropped irrelevant identifier columns.
      * Imputed missing values using median for numerical data and mode for categorical data.

2.  **Feature Engineering:**

      * Created a `customer_tenure_days` feature from the `joining_date` to represent how long each customer has been a member.

3.  **Exploratory Data Analysis (EDA):**

      * Visualized the relationships between key features and the churn risk to uncover initial patterns.

4.  **Modeling & Evaluation:**

      * The dataset was split into training (80%) and testing (20%) sets.
      * Categorical features were one-hot encoded and numerical features were standardized using `StandardScaler`.
      * Three models were trained and compared: **Logistic Regression**, **Random Forest**, and **XGBoost**.
      * The **XGBoost** model was identified as the best performer.

5.  **Model Interpretation & Explainability:**

      * Identified the most important features driving churn using the model's `feature_importances_`.
      * Conducted a **SHAP (SHapley Additive exPlanations)** analysis to understand not just which features are important globally, but how they influence individual customer predictions.

-----

## 📈 Results

The final, best-performing model was **XGBoost**, which achieved the following on the unseen test data:

  * **Accuracy:** **93.24%**
  * **Recall (for Churn class):** **95.4%** (Correctly identified 95.4% of all customers who actually churned).
  * **Precision (for Churn class):** **92.4%** (When it predicted churn, it was correct 92.4% of the time).

The SHAP analysis provides detailed visualizations, like the summary plot, which confirms the importance of loyalty and engagement features.

-----

## ⚙️ How to Use This Project

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install the required libraries.** Create a file named `requirements.txt` with the content below, then run the `pip install` command.

    **`requirements.txt`:**

    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    xgboost
    shap
    ```

    **Installation command:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the analysis:** The Python scripts in the repository contain the full workflow. Running the final script will also generate SHAP plots (`shap_summary_plot.png` and `shap_force_plot.html`).

-----

## 🔮 Future Work

This project serves as a strong foundation. Potential next steps include:

  * **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to find the optimal settings for the XGBoost model to improve accuracy further.
  * **Deployment:** Save the final trained model and create a simple API (e.g., using Flask or FastAPI) to serve predictions on new data.
  * **Model Monitoring:** Implement a system to monitor the model's performance over time in a production environment to detect concept drift.
