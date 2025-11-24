# Supply Chain Fraud & Late Delivery Classification

This repository contains code and data for **Advanced Statistics – Assignment 2** (Montpellier Business School).  
The goal is to build machine learning models to:

1. Detect **suspected fraud** orders  
2. Predict **late deliveries** using the `DataCoSupplyChainDataset.csv` supply chain dataset.

The main work is implemented in the **Stats2.ipynb** notebook (FINAL WORKING ASSIGNMENT CODE).

---

## 1. Repository Structure
```text
.
├─ Dashboard.py                  # (Existing) dashboard script – not used in this assignment
├─ DataCoSupplyChainDataset.csv  # Supply chain dataset
├─ Stats2.ipynb                  # Main assignment notebook (final working code)
├─ README.md                     # This file
├─ LICENSE                       # Project license
├─ .gitignore
└─ (optional IDE / venv folders)
```

---

## 2. Dataset

**File:** DataCoSupplyChainDataset.csv
**Description:** Transaction-level supply chain data for an international sporting goods company, including:
- Order and delivery dates
- Order status and delivery status
- Shipping mode and markets/regions
- Product and customer information
- Financial metrics (profit, sales, discounts, etc.)

**Target variables created in the notebook:**
- fraud
  - 1 if Order Status == "SUSPECTED_FRAUD"
  - 0 otherwise
- late_delivery
  - 1 if Delivery Status == "Late delivery"
  - 0 otherwise

---
 
## 3. Main Notebook: Stats2.ipynb

The notebook performs the full workflow:
**1. Load data**
```text
df = pd.read_csv("/content/DataCoSupplyChainDataset.csv", encoding="latin1")
```

**2. Create target variables** fraud and late_delivery.

**3. Select predictive features** (financial, shipping, product, and customer fields).

**4. Preprocessing**
- Label-encode categorical features
- Impute missing values with the median
- Handle class imbalance using SMOTE (oversampling the minority class)

**5. Modeling**
For each target (fraud, late_delivery) the notebook trains:
- Logistic Regression
- L1 Logistic Regression (sparse model with L1 penalty)
- Random Forest Classifier

**6. Evaluation**
For every model + target pair, the notebook reports:
- Accuracy
- Recall
- F1-score
- Full classification_report
- Confusion matrix plots

**7. Feature Importance**
- Trains RandomForestClassifier models for both targets
- Extracts and plots top features driving:
  - Fraud detection
  - Late delivery prediction

**8. Business Insights**
The notebook prints key insights and recommendations based on the most important features and model performance.

---

## 4. Dependencies

The project uses Python 3 and the following main libraries:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

In Colab, the only extra installation needed is:
```text
!pip install imbalanced-learn
```
Locally, you can install everything with:
```text
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

---

## 5. How to Run (Google Colab)

1. Open Stats2.ipynb in Google Colab (using the Open in Colab badge or by uploading it).
2. Upload DataCoSupplyChainDataset.csv to Colab so it appears under /content/.
3. Make sure the path in the notebook matches:
```text
df = pd.read_csv("/content/DataCoSupplyChainDataset.csv", encoding="latin1")
```
4. Run all cells from top to bottom:
- Package installation
- Data loading and preprocessing
- Model training and evaluation
- Feature importance plots and final results table

The notebook will output:
- Classification reports and confusion matrices for each model
- A combined performance table (Accuracy, Recall, F1 for both tasks)
- Top features for fraud and late delivery
- Text summary with business insights

---

## 6. How to Run Locally

1. Clone the repository:
```text
git clone https://github.com/<your-username>/SupplyChainDashboard.git
cd SupplyChainDashboard
```
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:
```text
pip install -r requirements.txt
```
(If you don’t have a requirements.txt, install the libraries listed in Section 4.)
4. Launch Jupyter or VS Code and open Stats2.ipynb.
5. Ensure the CSV path is correct (for local runs it will usually be):
```text
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")
```
6. Run all cells.

---

## 7. Results & Interpretation 

- Models are evaluated on both fraud detection and late delivery prediction.
- Performance is measured with Accuracy, Recall, and F1-score.
- The Random Forest and Logistic Regression models are compared to choose the best one for each task.
- Feature importance analysis identifies the top drivers of:
  - Order fraud (e.g., financial metrics, order value, discounts)
  - Late delivery (e.g., days for shipping, shipping mode, region)
    
- These insights can be used to:
  - Flag high-risk orders for manual review
  - Improve logistics planning and shipping strategies
  - Focus monitoring on the most influential risk factors
 
---

## 8. License

This project is shared under the license specified in the LICENSE file.

---

## 9. Acknowledgements

- Dataset and assignment context: Montpellier Business School – Advanced Statistics (Dr. Mehtab Alam Syed).
- Implementation and analysis: Student work for Assignment 2.
