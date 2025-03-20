# Predicting Which Telecom Customers Might Leave (Churn Prediction)

## What's This Project About?

This project is all about figuring out which customers of a telecom company are most likely to cancel their service (this is called "churn").  Knowing this in advance is super valuable because the company can then try to keep those customers by offering them special deals, better service, or other incentives. We're using machine learning to make these predictions.  Think of it like a "crystal ball" (though not quite as magical!) for customer retention.

## Where Did the Data Come From?

We used a publicly available dataset from Kaggle, a website where people share datasets for data science projects.  It's called the "Telco Customer Churn" dataset, and you can find it here: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset includes information about customers like:

*   **Who they are:** Things like whether they're a senior citizen, if they have a partner, and if they have dependents.
*   **What services they use:** Phone service, internet, online security, tech support, etc.
*   **Their contract:** What kind of contract they have (month-to-month, one year, two year), how they pay, and how much they pay.
*   **The important part:** Whether they *left* the company in the last month (this is what we're trying to predict).

## What Tools Did We Use?

We used Python, a popular programming language for data science, along with some helpful libraries (think of them as add-ons to Python):

*   **Pandas:** For handling and organizing the data.
*   **NumPy:** For doing math stuff.
*   **Scikit-learn:** This is the main machine learning library. It has tools for building and testing prediction models.
*   **Matplotlib and Seaborn:** For making charts and graphs to visualize the data.
*   **XGBoost:** A powerful type of machine learning model that often gives very good results.
*  **Google Colab:** to write and execute all the code.

## How Does It Work?

Here's a breakdown of what we did, step-by-step:

1.  **Get the Data Ready:**
    *   We loaded the data into the program.
    *   We looked at the data to understand it better (this is called "Exploratory Data Analysis" or EDA). We checked for missing information and made some charts to see patterns.
    *   We cleaned up the data. For example, we filled in some missing values for the "TotalCharges" column.
    *   We converted some of the text data (like "Yes" and "No") into numbers so the machine learning models could understand it.
    * We scaled some features.

2.  **Split the Data:** We split the data into two parts: a "training" set and a "testing" set.  We use the training set to teach the models, and the testing set to see how well they learned.

3.  **Build and Train the Models:** We used several different machine learning models:
    *   Logistic Regression (a simple, but often effective model)
    *   Decision Tree (like a flowchart of decisions)
    *   Random Forest (lots of decision trees working together)
    *   XGBoost (a more advanced technique)

4.  **See How Well the Models Did:** We tested each model on the "testing" data (data the models hadn't seen before). We used several metrics to see how good they were at predicting churn:
    *   **Accuracy:** How often did the model guess correctly overall?
    *   **Precision:** When the model predicted a customer would churn, how often was it right?
    *   **Recall:** Out of all the customers who *actually* churned, how many did the model correctly identify?
    *   **F1-score:** A balance between precision and recall.
    *   **ROC AUC:** A measure of how well the model can distinguish between customers who will churn and those who won't.

5. **Figure out the important features**: We checked which factors are most important when predicting if the user is going to churn or not.

## What Did We Find?

*(You'll need to fill this in with the *actual* results from running your code!  This is just an example.)*

| Model                 | Accuracy | Precision | Recall | F1-score | ROC AUC |
| --------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression   |   *Insert Results Here*  |    *Insert Results Here*       |  *Insert Results Here*      |     *Insert Results Here*     |   *Insert Results Here*      |
| Decision Tree         |     *Insert Results Here*     | *Insert Results Here*         |   *Insert Results Here*     |  *Insert Results Here*        |    -     |
| Random Forest         |   *Insert Results Here*      |    *Insert Results Here*       |  *Insert Results Here*      | *Insert Results Here*         |  *Insert Results Here*       |
| XGBoost               |    *Insert Results Here*     |     *Insert Results Here*      |   *Insert Results Here*     |    *Insert Results Here*      |  *Insert Results Here*       |

**Key Takeaways:**

*(Again, fill this in with *your* findings after you run the code.  This is just an example.)*

*   The XGBoost model seemed to do the best job overall at predicting churn.
*   The most important factors for predicting churn were the type of contract the customer had, their monthly charges, and how long they'd been a customer.  People on month-to-month contracts, with higher bills, and who were newer customers were more likely to leave.
*   We could probably make the model even better at finding *all* the customers who will churn (even if it means sometimes incorrectly predicting that someone will churn when they won't).
*   The telecom company could use these findings to try to keep customers. For example, they might offer discounts or better deals to people on month-to-month contracts to encourage them to sign up for longer-term plans.

## How to Run the Code Yourself

1.  Open the notebook file (`Telco_Churn_Prediction.ipynb`) in Google Colab (it's a free online tool from Google).
2.  Make sure you have the dataset file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) and upload it to Colab.
3.  Run each section of the code (each "cell") one by one.  Just click the little "play" button next to each cell.
4.  The results will be printed out as the code runs.

## What Could We Do Next?

*   **Try Different Settings:** We could try tweaking the settings of the models to see if we can make them even better.
*   **Add More Features:** We could try creating new data features from the existing ones.  For example, we could combine some features or create ratios.
*   **Handle the Imbalance Better:** The dataset has more customers who *didn't* churn than who did.  There are some special techniques we could use to handle this better.
*   **Make It a Web App:** We could turn this into a web application where someone could enter customer information and get a prediction of whether that customer is likely to churn.
*  **Use other algorithms:** Try using other machine learning algorithms like Support Vector Machines and neural networks.

## Who Made This?

Syed Shafin Ali
