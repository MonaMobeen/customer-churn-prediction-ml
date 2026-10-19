
# ConnectTel Customer Churn Prediction

**ML-01: "Decision Tree and Random Forest"**

This project uses Machine Learning to predict whether a telecom customer is likely to leave a service provider. It implements and compares two classification algorithms: Decision Tree and Random Forest.

## Project Objective

Customer churn occurs when a customer stops using a company's services.

The goal is to predict customer churn using information such as contract type, tenure, monthly charges, payment method, and subscribed services.

## Dataset

- **Dataset:** Telco Customer Churn
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **CSV file:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Total records:** 7,043 customers
- **Total columns:** 21
- **Target variable:** `Churn`
- **Target classes:** `Yes` and `No`

The dataset contains customer profiles, account details, subscribed services, billing information, and churn status.

**Note:** The dataset is not included in this repository. Download it from Kaggle before running the notebook.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn

## ML Algorithms

### 1. Decision Tree Classifier

A Decision Tree learns patterns by splitting data according to feature values. It uses these patterns to predict whether a customer will leave.

### 2. Random Forest Classifier

A Random Forest combines predictions from multiple Decision Trees to produce a final prediction. It can learn different patterns from the dataset.

## Project Workflow

1. Load the customer churn dataset.
2. Explore the dataset and understand its features.
3. Handle missing values and preprocess the data.
4. Separate input features (`X`) and the target (`y`).
5. Convert categorical variables into numerical features.
6. Split the data into training and testing sets.
7. Train the Decision Tree and Random Forest models.
8. Generate predictions on the test dataset.
9. Evaluate and compare both models.

## What the Notebook Covers (`churn_final.ipynb`)
 
1. **Problem understanding:** what churn is, cost of false negatives vs false positives, hypotheses, why `customerID` and `Churn` are not inputs, and what the data can and cannot establish.
2. **Manual split calculations:** Gini and entropy by hand for an 8-customer table, choosing the root split, tree terminology.
3. **Data audit and leakage-free setup:** 11 blank `TotalCharges` values (all tenure 0) turned into missing values and imputed inside the pipeline; no duplicates; stratified 60% train / 20% validation / 20% sealed test split (`random_state=42`); a dummy baseline.
4. **Decision Tree:** unrestricted tree, depth curve, `min_samples_leaf` / `min_samples_split`, Gini vs entropy, cost-complexity pruning, a readable depth-3 tree, tracing two customers, stability over three splits.
5. **Random Forest:** number of trees, `max_features`, bootstrap, out-of-bag score, per-tree averaging, stability over three splits.
6. **Imbalance, tuning and selection:** `class_weight`, `RandomizedSearchCV` for both models, candidate comparison on validation, threshold choice (0.3 / 0.5 / 0.7), permutation importance.
7. **Final test evaluation and error analysis:** the selected model is refit on train + validation and scored once on the sealed test set.
8. **Regression extension and deployment:** trees/forests for `MonthlyCharges`; the final pipeline is saved with `joblib`, with a metadata file and a prediction function that validates its input columns.
## Model Configuration

| Parameter | Decision Tree | Random Forest |
|---|---|---|
| Algorithm | DecisionTreeClassifier | RandomForestClassifier |
| Maximum depth | 5 | 5 |
| Number of trees | N/A | 100 |
| Random state | 42 | 42 |

## Results

Both models were evaluated on the same test set containing **1,407 customers**.

### Model Comparison

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 77.83% | 79.46% |
| Churn Precision (`Yes`) | 58.1% | 68.4% |
| Churn Recall (`Yes`) | 59.6% | 42.2% |
| Churn F1-score (`Yes`) | 58.8% | 52.2% |

### Results 
 
### Validation comparison (candidates)
 
| Candidate | F1 | Precision | Recall | Avg. precision |
|---|---:|---:|---:|---:|
| Dummy (always "No") | 0.000 | 0.000 | 0.000 | 0.265 |
| Tuned tree | 0.614 | 0.498 | 0.800 | 0.594 |
| Pruned tree (best `ccp_alpha`) | 0.535 | 0.595 | 0.487 | 0.517 |
| **Tuned forest (selected)** | **0.624** | 0.523 | 0.775 | **0.639** |
 
Selected model: Random Forest with `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5`, `max_features="sqrt"`, `class_weight="balanced"`. Threshold 0.5 gave the best validation F1 (0.5743 at 0.3, 0.6243 at 0.5, 0.5740 at 0.7).
 
### Final result on the sealed test set (1,409 customers)
 
| Metric | Value |
|---|---:|
| Accuracy | 75.4% |
| Churn precision | 52.5% |
| Churn recall | 77.3% |
| Churn F1 | 62.5% |
| ROC-AUC | 0.843 |
| Average precision | 0.653 |

**Decision Tree**
- Achieved 77.83% accuracy.
- Identified approximately 59.6% of customers who actually churned.
- Achieved a higher churn recall and F1-score than Random Forest.

**Random Forest**
- Achieved 79.46% accuracy.
- Improved accuracy by approximately 1.63 percentage points.
- Achieved higher churn precision at 68.4%.
- Had lower churn recall at 42.2%, meaning it missed more actual churners.
Confusion matrix (rows = actual Stay/Churn, columns = predicted Stay/Churn): `[[773, 262], [85, 289]]`, so the model catches 289 of 374 real churners and wrongly flags 262 customers who stayed.
 
### Interpretation
 
- Contract type is by far the most important input (permutation importance 0.103 F1 drop), followed by InternetService (0.031) and tenure (0.029).
- Accuracy is lower than an untuned, unweighted forest would show because `class_weight="balanced"` deliberately trades accuracy for recall. A model that predicts "No" for everyone already reaches about 73% accuracy and finds no churners, so accuracy alone is misleading here.
- A single tree is unstable (validation F1 std 0.038 over three splits) while the forest is steadier (std 0.013).
- Regression extension: a Random Forest for `MonthlyCharges` reached test MAE 0.99, RMSE 1.38, R2 0.998. This is high because the price is largely determined by the subscribed services.
### Which Model Performed Better?

Random Forest achieved higher overall accuracy, while Decision Tree performed better at identifying customers who actually churned.

If the main objective is to identify as many potential churners as possible, Decision Tree is the stronger choice based on churn recall and F1-score in this experiment.

## Key Concepts Learned

- Supervised Machine Learning
- Classification
- Decision Tree learning
- Ensemble learning with Random Forest
- Data preprocessing and categorical encoding
- Train-test splitting
- Accuracy, precision, recall, and F1-score
- Model evaluation and comparison
- Class imbalance
- Business interpretation of model results

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/MonaMobeen/customer-churn-prediction-ml
```


### Step 2: Install Dependencies

Ensure Python and Jupyter Notebook are installed, then run:

```bash
pip install pandas numpy scikit-learn jupyter
```

### Step 3: Download the Dataset

Download the CSV file from [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Place it in the location expected by your notebook.

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

Open your project notebook and verify that the dataset path matches your CSV file location.

### Step 5: Run the Notebook

Execute the notebook cells from top to bottom to preprocess the data, train both models, generate predictions, and compare their performance.

## Saved Model
 
- `churn_pipeline.joblib` - preprocessing + tuned forest, fitted on train + validation
- `churn_metadata.json` - threshold, model name, seed, scikit-learn version, required input columns
- `test_predictions.csv` - customerID, actual label, churn probability and prediction for the test set
- `results_table.csv` - every experiment run in the notebook
Only load `.joblib` files from sources you trust. The model was saved with scikit-learn 1.9.1, so use the same version (see `requirements.txt`).

## Limitations

- Predictions are not always correct.
- Accuracy alone does not show how well a model identifies churners.
- Random Forest missed more actual churners than Decision Tree in this experiment.
- Results depend on the dataset, preprocessing, model settings, and train-test split.
- The dataset reveals patterns associated with churn but does not prove why customers leave or which retention strategy will work.

## Conclusion

This project demonstrates how Machine Learning can help analyze and predict customer churn using telecom customer data.

Random Forest achieved the highest accuracy of **79.46%**, while Decision Tree achieved a higher churn recall of **59.6%** and an F1-score of **58.8%**.

The experiment demonstrates why model selection should consider multiple evaluation metrics and business objectives rather than accuracy alone.

  

