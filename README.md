# HOUSE-PRICE-ADVANCED-MACHINE-LEARNING-
The objective was to compare different machine learning models and determine which model performs best for predicting house prices.

## 📌 Project Overview
This repository contains the implementation, evaluation and structural optimization of advanced machine learning pipelines applied to residential real estate valuation. Shifting from simple baseline estimations into advanced, non-linear ensemble frameworks, this project benchmarks Decision Trees, Random Forests, and Gradient Boosting Regressors to map out continuous housing asset trends. 

The analytical workflow systematically balances the bias-variance tradeoff by utilizing `GridSearchCV` cross-validation to constrain tree architectures against localized overfitting.

## 📊 Performance Comparison Matrix
The structural performance breakdown across all five experimental iterations is detailed below:

| Machine Learning Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Coefficient of Determination (R² Score) |
| :--- | :---: | :---: | :---: |
| **Baseline (Linear Reg)** | 1,029,010.29 | 1,381,117.65 | **0.622622** |
| **Decision Tree** | 1,358,952.60 | 1,842,739.80 | 0.328194 |
| **Random Forest** | 1,065,957.23 | 1,443,776.77 | 0.587603 |
| **Gradient Boosting** | 1,040,051.69 | 1,439,009.70 | 0.590321 |
| **Tuned Gradient Boosting** | 1,064,053.94 | 1,433,722.45 | **0.593326** |

## 💡 Key Insights & Technical Findings
* **The Overfitting Vulnerability:** An unconstrained standalone Decision Tree experienced severe high-variance degradation ($R^2 = 0.328194$). Without branch limits, it memorized data noise rather than structural patterns, leading to heavy errors on test partitions.
* **Ensemble Optimization:** Random Forest and Gradient Boosting architectures successfully neutralized individual tree volatility, immediately elevating variance explanation metrics back to the ~59% boundaries.
* **Hyperparameter Optimization Impact:** Running `GridSearchCV` to optimize maximum tree depth, split boundaries, and learning iterations successfully dropped the Gradient Boosting error envelope down to an RMSE of **1,433,722.45** and raised the $R^2$ score to **0.593326**.
* **Linear Dominance Phenomenon:** The baseline Linear Regression model maintained the overall highest variance explanation ($R^2 = 0.622622$). This indicates that the engineered feature space (area, bathrooms, stories, etc.) shares a predominantly linear relationship with market pricing, enabling low-complexity hyperplanes to generalize highly effectively on this specific data allocation.

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.x
* **Core Frameworks:** Scikit-Learn, Pandas, NumPy
* **Visualization Stack:** Matplotlib, Seaborn

