# Heart Disease Data Preprocessing & EDA

An exploratory data analysis and data preprocessing project using a Heart Disease dataset.

The goal of this project is to understand the dataset, identify data-quality issues, explore relationships between features, and transform the raw data into a model-ready format.

## Project Overview

Before building a machine learning model, raw data needs to be properly understood and prepared.

In this project, I worked through the following pipeline:

**Raw Data → EDA → Data Cleaning → Encoding → Feature Scaling → Model-Ready Data**

## Dataset

The dataset contains patient-related health information and a target variable:

* `Age` — Age of the patient
* `Sex` — Sex of the patient
* `ChestPainType` — Type of chest pain
* `RestingBP` — Resting blood pressure
* `Cholesterol` — Cholesterol level
* `FastingBS` — Fasting blood sugar
* `RestingECG` — Resting ECG results
* `MaxHR` — Maximum heart rate achieved
* `ExerciseAngina` — Exercise-induced angina
* `Oldpeak` — ST depression
* `ST_Slope` — Slope of the peak exercise ST segment
* `HeartDisease` — Target variable

## Exploratory Data Analysis

The following analysis was performed:

* Checked dataset shape and structure
* Examined data types using `info()`
* Generated descriptive statistics
* Checked duplicate records
* Checked missing values
* Analyzed target distribution
* Visualized numerical distributions using histograms
* Used KDE to understand feature distributions
* Compared categorical variables with the target
* Analyzed distributions using boxplots and violin plots
* Examined numerical correlations using a heatmap

## Data Cleaning

During the analysis, some `0` values were found in `Cholesterol` and `RestingBP`.

Instead of treating these values as valid measurements, they were replaced with the mean of the non-zero observations.

```python
ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean)

resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean)
```

The values were then rounded to two decimal places.

## Categorical Encoding

Categorical variables were converted into numerical features using one-hot encoding.

```python
df_encode = pd.get_dummies(df, drop_first=True)
df_encode = df_encode.astype(int)
```

This allows categorical information to be represented numerically for machine learning algorithms.

## Feature Scaling

Standardization was applied to the numerical features:

* `Age`
* `RestingBP`
* `Cholesterol`
* `MaxHR`
* `Oldpeak`

Using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

scaler = StandardScaler()
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])
```

Standardization transforms the features so that they have a mean close to 0 and a standard deviation close to 1.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

## Project Structure

```text
heart-disease-data-preprocessing/
│
├── data/
│   └── heart.csv
│
├── notebooks/
│   └── heart_disease_data_preprocessing.ipynb
│
├── README.md
└── requirements.txt
```

## Key Learning

This project helped me understand that machine learning is not only about selecting an algorithm.

A significant part of the workflow involves:

**Understanding the data → Finding data-quality issues → Cleaning → Transforming → Scaling → Preparing for modeling**

This project focuses on the data preparation stage before machine learning model development.

## Future Work

The next step would be to use the processed dataset to:

* Split the data into training and testing sets
* Train classification models
* Evaluate model performance
* Compare different machine learning algorithms
* Perform feature selection
* Tune model hyperparameters

## Author

**Ritesh Das**

Aspiring Data / Analytics Professional
