import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("dataset/preprocess_dataset3.csv")  # replace with your filename

# Step 2: Clean & preprocess
# Create average salary as target
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# Select features (you can add more if needed)
features = ['job_title', 'category', 'role', 'location', 'type']
target = 'avg_salary'

# Drop rows with missing values in selected columns
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Step 3: Encode categorical features
categorical_features = features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 4: Build the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("✅ Model Evaluation:")
print(f"MAE: RM{mae:.2f}")
print(f"RMSE: RM{rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 8: Save model (optional)
joblib.dump(model, 'salary_predictor_model.pkl')

# Predict example
sample_input = pd.DataFrame([{
    'job_title': 'Accounts Executive',
    'category': 'Accounting',
    'role': 'accounts-executive',
    'location': 'Petaling',
    'type': 'Full time'
}])

predicted_salary = model.predict(sample_input)[0]
print(f"\n📌 Predicted Average Salary for input: RM{predicted_salary:.2f}")
