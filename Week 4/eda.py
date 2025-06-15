import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('train.csv')  # adjust filename/path as needed

# 2. Quick peek
print("\n=== First 5 rows ===")
print(df.head())
print("\n=== Data Info ===")
print(df.info())
print("\n=== Descriptive Statistics (Numeric) ===")
print(df.describe())
print("\n=== Descriptive Statistics (Categorical) ===")
print(df.describe(include=['object', 'category']))

# 3. Missing-value summary
print("\n=== Missing Values per Column ===")
print(df.isnull().sum())

# 4. Outlier detection (IQR method) for numeric columns
numeric_cols = ['Age', 'Fare']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\nColumn '{col}' → {len(outliers)} outliers (IQR method)")

# 5. Correlation matrix heatmap (only numeric)
plt.figure(figsize=(6,5))
corr = df.select_dtypes(include=['int64','float64']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 6. Distributions (histograms)
for col in numeric_cols:
    plt.figure()
    df[col].hist(bins=30, edgecolor='k')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# 7. Box plots (visual outliers)
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()

# 8. Scatter plot: Age vs. Fare
plt.figure()
plt.scatter(df['Age'], df['Fare'], alpha=0.6)
plt.title('Scatter Plot: Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.tight_layout()
plt.show()

# 9. Categorical relationships
#   a) Survival rate by Sex
surv_sex = df.groupby('Sex')['Survived'].mean()
plt.figure()
surv_sex.plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.show()

#   b) Survival rate by Pclass
surv_class = df.groupby('Pclass')['Survived'].mean()
plt.figure()
surv_class.plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.show()

# 10. Missing-value heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap (True = Missing)')
plt.xlabel('Columns')
plt.ylabel('Records')
plt.tight_layout()
plt.show()
