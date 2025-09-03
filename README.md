# Customer Churn Prediction Project 📊

This project is a complete data science workflow aimed at predicting customer churn using a provided dataset. The primary goal is to build a reliable machine learning model that can identify customers who are likely to stop using a service, enabling proactive retention strategies.

The project progresses from initial data cleaning and exploratory analysis to building, comparing, and interpreting multiple machine learning models. The final result is a highly accurate **Random Forest** model with an accuracy of over 93%.

## 🚀 Key Insights & Findings

Our analysis revealed several key factors that strongly influence customer churn:

  * **Loyalty Program Engagement is Critical:** The most significant predictor of churn is a customer's engagement with the loyalty program. Features like **`points_in_wallet`** and **`membership_category`** are the most important factors.
  * **Behavioral Metrics Matter:** User activity, such as **`avg_transaction_value`**, **`avg_time_spent`**, and how recently they logged in, are also powerful predictors of churn.
  * **Negative Feedback is a Major Red Flag:** Customers providing feedback like "Poor Customer Service" or "Poor Product Quality" have a significantly higher churn risk.

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

      * Visualized the relationships between features like membership category, customer tenure, and feedback against the churn risk.

4.  **Modeling & Evaluation:**

      * The dataset was split into training (80%) and testing (20%) sets.
      * Categorical features were one-hot encoded and numerical features were standardized using `StandardScaler`.
      * Two models were trained and compared: **Logistic Regression** (baseline) and **Random Forest**.
      * The **Random Forest** model was identified as the best performer.

5.  **Interpretation:**

      * Used the best model's **feature importances** to identify the key drivers of churn.

-----

## 📈 Results

The final, best-performing model was the **Random Forest Classifier**, which achieved the following on the unseen test data:

  * **Accuracy:** **93.03%**
  * **Recall (for Churn class):** **95%** (Correctly identified 95% of all customers who actually churned).
  * **Precision (for Churn class):** **92%** (When it predicted churn, it was correct 92% of the time).

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

4.  **Run the analysis:** The Python scripts in the repository contain the full workflow.

-----

## 🔮 Future Work

This project serves as a strong foundation. Potential next steps include:

  * **Hyperparameter Tuning:** Use `GridSearchCV` to find the optimal settings for the Random Forest model to potentially improve accuracy further.
  * **Explainable AI (XAI):** Use libraries like SHAP to explain individual predictions, providing deeper, customer-level insights.
  * **Deployment:** Save the final trained model and create a simple API (e.g., using Flask or FastAPI) to serve predictions on new data.

