import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv("dataset/preprocessed_dataset.csv")

# ========== BASIC INFO ========== #
print("📄 Dataset Info:")
print(df.info())

print("\n🧼 Missing Values:")
print(df.isnull().sum())

print("\n🔢 Summary Statistics:")
print(df.describe(include='all'))

# ========== SALARY PREP ========== #
# Calculate avg_salary if not already done
if 'avg_salary' not in df.columns and 'min_salary' in df.columns and 'max_salary' in df.columns:
    df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# ========== 1. SALARY DISTRIBUTION ========== #
plt.figure(figsize=(10, 6))
sns.histplot(df['avg_salary'], kde=True, bins=40, color='skyblue')
plt.title("Distribution of Average Salary")
plt.xlabel("Average Salary (RM)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ========== 2. SALARY BY LOCATION ========== #
plt.figure(figsize=(12, 6))
top_locations = df['location'].value_counts().head(10).index
sns.boxplot(data=df[df['location'].isin(top_locations)], x='location', y='avg_salary')
plt.title("Salary Distribution by Top 10 Job Locations")
plt.xticks(rotation=45)
plt.ylabel("Average Salary (RM)")
plt.tight_layout()
plt.show()

# ========== 3. SALARY BY CATEGORY ========== #
plt.figure(figsize=(12, 6))
top_categories = df['category'].value_counts().head(10).index
sns.boxplot(data=df[df['category'].isin(top_categories)], x='category', y='avg_salary')
plt.title("Salary Distribution by Job Category")
plt.xticks(rotation=45)
plt.ylabel("Average Salary (RM)")
plt.tight_layout()
plt.show()

# ========== 4. SALARY BY ROLE ========== #
plt.figure(figsize=(12, 6))
top_roles = df['role'].value_counts().head(10).index
sns.boxplot(data=df[df['role'].isin(top_roles)], x='role', y='avg_salary')
plt.title("Salary Distribution by Job Role")
plt.xticks(rotation=45)
plt.ylabel("Average Salary (RM)")
plt.tight_layout()
plt.show()

# ========== 5. JOB TYPE DISTRIBUTION ========== #
plt.figure(figsize=(7, 4))
sns.countplot(x='type', data=df, order=df['type'].value_counts().index, palette='Set2')
plt.title("Job Type Distribution")
plt.xlabel("Employment Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ========== 6. CORRELATION HEATMAP (Optional for numerical) ========== #
plt.figure(figsize=(6, 4))
numerical_cols = ['min_salary', 'max_salary', 'avg_salary']
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Between Salary Fields")
plt.tight_layout()
plt.show()
