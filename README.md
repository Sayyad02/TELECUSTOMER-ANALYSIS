# Customer Churn Prediction with Machine Learning

## Project Overview

This project aims to predict customer churn for a telecommunications company using machine learning techniques.  Customer churn, the loss of customers, is a critical business problem, and predicting which customers are at risk of churning allows companies to take proactive steps to retain them.  This project uses a publicly available dataset from Kaggle to build, train, and evaluate several classification models.

## Dataset

The dataset used in this project is the Telco Customer Churn dataset from Kaggle: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains information about customer demographics, services used, contract details, and whether the customer churned (left the company) within the last month.  Key features include:

*   **Demographics:** `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`
*   **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
*   **Contract & Billing:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
*   **Target Variable:** `Churn` (Yes/No)

## Tools and Technologies

*   **Programming Language:** Python 3.x
*   **Libraries:**
    *   Pandas: Data manipulation and analysis.
    *   NumPy: Numerical operations.
    *   Scikit-learn: Machine learning algorithms, model evaluation, and preprocessing.
    *   Matplotlib and Seaborn: Data visualization.
    *   XGBoost: Gradient boosting library.
*   **Development Environment:** Google Colab (cloud-based Jupyter Notebook environment).

## Project Structure

The project is implemented in a Jupyter Notebook (`Telco_Churn_Prediction.ipynb`) and follows these main steps:

1.  **Data Loading and Exploration:**
    *   Load the dataset into a Pandas DataFrame.
    *   Perform Exploratory Data Analysis (EDA) to understand the data's characteristics, identify missing values, and visualize distributions and relationships between features.

2.  **Data Preprocessing:**
    *   Handle missing values in `TotalCharges` by imputing based on `tenure` groups.
    *   Convert `TotalCharges` to a numerical data type.
    *   One-hot encode categorical features using `OneHotEncoder`.
    *   Scale numerical features using `StandardScaler`.
    *   Create a preprocessing pipeline using `ColumnTransformer` and `Pipeline` to combine these steps.

3.  **Data Splitting:**
    *   Split the data into training and testing sets (80% training, 20% testing) using `train_test_split` with stratification to maintain the churn/no-churn ratio.

4.  **Model Building and Training:**
    *   Build and train the following classification models using scikit-learn and XGBoost:
        *   Logistic Regression
        *   Decision Tree
        *   Random Forest
        *   XGBoost
    *   Create separate pipelines for each model, combining the preprocessing steps with the model itself.

5.  **Model Evaluation:**
    *   Evaluate the performance of each model on the test set using the following metrics:
        *   Accuracy
        *   Precision
        *   Recall
        *   F1-score
        *   Confusion Matrix
        *   ROC AUC (and plot ROC curve)

6.  **Feature Importance (Optional):**
    *   Analyze feature importances for tree-based models (Random Forest and XGBoost) to identify the most influential predictors of churn.

## Results

The following table summarizes the performance of the different models on the test set:
| Model                 | Accuracy | Precision | Recall | F1-score | ROC AUC |
| --------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression   |   *Insert Results Here*  |    *Insert Results Here*       |  *Insert Results Here*      |     *Insert Results Here*     |   *Insert Results Here*      |
| Decision Tree         |     *Insert Results Here*     | *Insert Results Here*         |   *Insert Results Here*     |  *Insert Results Here*        |    -     |
| Random Forest         |   *Insert Results Here*      |    *Insert Results Here*       |  *Insert Results Here*      | *Insert Results Here*         |  *Insert Results Here*       |
| XGBoost               |    *Insert Results Here*     |     *Insert Results Here*      |   *Insert Results Here*     |    *Insert Results Here*      |  *Insert Results Here*       |

**Replace the placeholders above ( *Insert Results Here* ) with the actual results you obtained from running your notebook.**  Copy and paste the output from your `evaluate_model` function for each model.  Be sure to *actually run the notebook* and get results *before* filling in this table.

**Key Findings:**

*   *(Write a summary of your findings here.  For example:)*
    *   Which model performed best, and why do you think that is?
    *   Which features were most important in predicting churn?
    *   Are there any surprising or unexpected results?
    *   What are the limitations of the analysis?
    *  What business insights can be used by the telecom company?

**Example Key Findings (Replace this with YOUR findings):**

*   The XGBoost model achieved the highest F1-score and ROC AUC, indicating the best overall performance in predicting churn. This is likely due to XGBoost's ability to handle complex interactions between features.
*   Feature importance analysis revealed that contract type, monthly charges, and tenure were among the most important predictors of churn. Customers on month-to-month contracts, with higher monthly charges, and shorter tenures were more likely to churn.
*   The model's recall score (ability to identify *all* customers who will churn) could be further improved. This is important for minimizing false negatives (customers who churn but were not predicted to).
* The telecom company should consider offering incentives to customers on month-to-month contracts to switch to longer-term contracts, potentially reducing churn rates. They may also want to investigate the reasons for higher churn among customers with higher monthly charges.

## How to Run the Code

1.  **Open the Notebook:** Open the `Telco_Churn_Prediction.ipynb` notebook in Google Colab ([https://colab.research.google.com/](https://colab.research.google.com/)).
2.  **Upload the Dataset:** Follow the instructions in the notebook (Section 1) to upload the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file to your Colab session.
3.  **Run the Cells:** Execute each code cell in the notebook sequentially by clicking the "Play" button (or pressing Shift+Enter).
4.  **View Results:** The notebook will output the results of the data analysis, model training, and evaluation.

## Future Improvements

*   **Hyperparameter Tuning:**  Experiment with different hyperparameters for each model (e.g., using `GridSearchCV` or `RandomizedSearchCV`) to optimize performance.
*   **Feature Engineering:**  Create new features from the existing ones (e.g., combining features, creating interaction terms) to potentially improve model accuracy.
*   **Handle Imbalanced Data More Effectively:** Explore more advanced techniques for handling imbalanced datasets (e.g., SMOTE, ADASYN).
*   **Deployment:** Deploy the best-performing model as a web service (e.g., using Flask or FastAPI) to make predictions on new data.
* **Try other models:** Use other models like SVM and neural networks.

## Author

Syed Shafin Ali
