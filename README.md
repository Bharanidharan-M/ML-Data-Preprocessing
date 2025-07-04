# ML-Data-Preprocessing
# Employee Dataset Preprocessing

This project demonstrates **data preprocessing techniques** on a small employee dataset using **Pandas**, **Scikit-learn**, **Seaborn**, and **Missingno**.

## 📂 Dataset

- **File:** `preprocessing_dataset.csv`
- **Columns:** 
  - `Name`
  - `Age`
  - `Gender`
  - `Salary`
  - `Department`

## 🛠️ Techniques Used

✅ **Handling Missing Values**
- Used `dropna` to remove rows with missing values.
- Alternatively filled:
  - `Age` → median
  - `Salary` → mean
  - `Gender` → mode

✅ **Outlier Detection**
- **IQR Method**
- **Z-score**
- **Isolation Forest**

✅ **Categorical Encoding**
- `LabelEncoder` for `Gender`
- `get_dummies` + `OneHotEncoder` for `Department`

✅ **Feature Scaling**
- **Standardization** with `StandardScaler` → `Age_std`, `Salary_std`
- **Normalization** with `MinMaxScaler` → `Age_norm`, `Salary_norm`

✅ **Missing Value Visualization**
- `missingno.bar`
- `missingno.matrix`
- `sns.heatmap(data.isnull())`

✅ **Outlier Visualization**
- Box plot for `Age`
- Scatter plot for `Age`

## 📈 Example Outputs

- **Isolation Forest outlier:**  
  ```
  Name   Age  Gender  Salary  Department  outlier
  David  40.0  M      58000.0  HR         -1
  ```

- **Standardized / Normalized Columns:**  
  | Name  | Age_std | Salary_std | Age_norm | Salary_norm |
  |--------|---------|------------|----------|-------------|
  | Alice | -0.71   | -1.69       | 0.17     | 0.00        |
  | Bob   | 0.25    | 0.50        | 0.44     | 0.77        |
  | David | 2.16    | 0.06        | 1.00     | 0.62        |
  | ...   | ...     | ...         | ...      | ...         |

## 🚀 How to Run

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

# Load data
data = pd.read_csv("/content/preprocessing_dataset.csv")

# Handle missing values
data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Salary"].fillna(data["Salary"].mean(), inplace=True)

# Encode categorical
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data = pd.get_dummies(data, columns=["Department"])

# Outlier detection
iso = IsolationForest(contamination=0.1)
data["outlier"] = iso.fit_predict(data[["Age", "Salary"]])

# Scaling
scaler = StandardScaler()
data["Age_std"] = scaler.fit_transform(data[["Age"]])
data["Salary_std"] = scaler.fit_transform(data[["Salary"]])

minmax = MinMaxScaler()
data["Age_norm"] = minmax.fit_transform(data[["Age"]])
data["Salary_norm"] = minmax.fit_transform(data[["Salary"]])
```

## 📎 References

- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Missingno](https://github.com/ResidentMario/missingno)

---

✅ *Tip:* Use [nbviewer](https://nbviewer.org/) for better rendering if viewing this notebook on GitHub.
