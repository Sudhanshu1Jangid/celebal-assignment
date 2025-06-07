import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = pd.read_csv("path_to_titanic.csv")

# Visualizing the distribution of the 'Age' feature
plt.figure(figsize=(10,6))
sns.histplot(df['Age'], kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Titanic Dataset')
plt.show()

# Visualizing the count of passengers by class
plt.figure(figsize=(8,6))
sns.countplot(x='Pclass', data=df, palette='pastel')
plt.title('Count of Passengers by Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Visualizing survival rate by sex
plt.figure(figsize=(8,6))
sns.countplot(x='Survived', hue='Sex', data=df, palette='muted')
plt.title('Survival Rate by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Visualizing the relationship between age and fare
plt.figure(figsize=(10,6))
sns.scatterplot(x='Age', y='Fare', data=df, hue='Survived', palette='coolwarm')
plt.title('Age vs Fare (Survival Status)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()
