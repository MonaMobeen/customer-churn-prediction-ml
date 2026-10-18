
# Customer Churn Prediction Using Machine Learning

## Project Overview
This project predicts whether a telecom customer is likely to leave the company's services using Machine Learning classification algorithms.

## Dataset
- **Dataset:** Telco Customer Churn
- **Total Records:** 7,043
- **Total Columns:** 21
- **Target Column:** `Churn`
- **Features:** Customer information, services, contract details, and billing information.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

## Project Workflow
1. Load the dataset.
2. Explore the data and check missing values and duplicates.
3. Convert `TotalCharges` into numeric format.
4. Handle missing values.
5. Remove the `customerID` column.
6. Separate features (`X`) and target (`y`).
7. Convert categorical variables into numerical values using one-hot encoding.
8. Split the dataset into 80% training data and 20% testing data.
9. Train a Decision Tree classifier.
10. Train a Random Forest classifier.
11. Evaluate and compare both models.

## Machine Learning Models

### 1. Decision Tree
A Decision Tree learns rules from customer data to predict whether a customer will leave the service.

- Accuracy: **77.83%**
- Recall for Churn = Yes: **60%**
- F1-score for Churn = Yes: **59%**

### 2. Random Forest
Random Forest combines multiple decision trees to make predictions.

- Accuracy: **79.46%**
- Recall for Churn = Yes: **42%**
- F1-score for Churn = Yes: **52%**

## Model Comparison

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 77.83% | 79.46% |
| Churn Precision | 58% | 68% |
| Churn Recall | 60% | 42% |
| Churn F1-score | 59% | 52% |

*Note: Churn precision, recall, and F1-score are rounded from the classification reports.*

## Results and Conclusion
Random Forest achieved higher overall accuracy than Decision Tree. However, Decision Tree performed better at identifying customers who actually churned, with 60% recall compared with 42% for Random Forest.

Therefore, accuracy alone is not enough to select the best model. Recall and F1-score are also important because identifying customers who may leave is a key objective of churn prediction.

This project demonstrates data preprocessing, feature encoding, model training, prediction, and evaluation using Python and Scikit-learn.

## How to Run
1. Install Python and Jupyter Notebook.
2. Install the required libraries:

   ```bash
   pip install pandas scikit-learn notebook
   ```

3. Place the CSV dataset in the `data` folder.
4. Open the project notebook in Jupyter Notebook.
5. Run the cells in order.

## Project Structure

```text
Customer-Churn-Prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_prediction.ipynb
└── README.md
```

*Note: Update the notebook filename in this structure if your actual filename is different.*
