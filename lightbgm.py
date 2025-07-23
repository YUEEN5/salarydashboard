import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import joblib

# Step 1: Load the dataset
df = pd.read_csv("dataset/preprocess_dataset3.csv")  # Replace with your file

# Step 2: Create average salary
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# Step 3: Select features
features = ['job_title', 'category', 'role', 'location', 'type']
target = 'avg_salary'

# Drop missing values in relevant columns
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Step 4: Preprocess categorical variables
categorical_features = features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 5: Build LightGBM pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("✅ LightGBM Model Evaluation:")
print(f"MAE: RM{mae:.2f}")
print(f"RMSE: RM{rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Save model
joblib.dump(model, 'salary_predictor_lgbm.pkl')

# Step 10: Predict example
sample_input = pd.DataFrame([{
    'job_title': 'Accounts Executive',
    'category': 'Accounting',
    'role': 'accounts-executive',
    'location': 'Petaling',
    'type': 'Full time'
}])

predicted_salary = model.predict(sample_input)[0]
print(f"\n📌 Predicted Average Salary: RM{predicted_salary:.2f}")
