import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import numpy as np

# Load dataset
df = pd.read_csv("dataset/clean_preprocessed_dataset.csv")

df['log_mean_salary'] = np.log1p(df['mean_salary'])  # log(1 + x)

# Drop irrelevant columns
df = df.drop(columns=["job_id", "salary", "min_salary", "max_salary", "listingDate"])

# Define features and target
selected_features = ['category', 'role', 'location', 'type']
X = df[selected_features]
y = df['log_mean_salary']

# Split data
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
cat_features = [col for col in X.columns if X[col].dtype == 'object']

# Initialize and train model
model_cb = CatBoostRegressor(verbose=0, random_state=42)
model_cb.fit(X_train_cat, y_train_cat, cat_features=cat_features)

# Save the model
model_cb.save_model("catboost_salary_model2.cbm")

# Evaluate the model
# y_pred_cb = model_cb.predict(X_test_cat)

# 1) After training and predicting on log-scale:
y_pred_log = model_cb.predict(X_test_cat)

# 2) Inverse-transform both predictions and true values:
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test_cat)


print("CatBoost MAE:", mean_absolute_error(y_test_orig, y_pred_orig))
print("CatBoost R²:", r2_score(y_test_orig, y_pred_orig))
print("CatBoost RMSE:", mean_squared_error(y_test_orig, y_pred_orig) ** 0.5)

from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

model = CatBoostRegressor(verbose=0, random_state=42)

param_grid = {
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [500, 1000],
    'l2_leaf_reg': [3, 5, 7]
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error')
grid.fit(X_train_cat, y_train_cat, cat_features=cat_features)

print("Best parameters:", grid.best_params_)
