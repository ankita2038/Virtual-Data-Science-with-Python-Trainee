# Week 1: Data Acquisition, Cleaning and Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load public dataset
df = pd.read_csv("train.csv")

# 2. Initial exploration
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. Create a copy before cleaning
df_clean = df.copy()

# 4. Handle missing values
df_clean.drop("Cabin", axis=1, inplace=True)

median_age = df_clean["Age"].median()
df_clean["Age"] = df_clean["Age"].fillna(median_age)

mode_embarked = df_clean["Embarked"].mode()[0]
df_clean["Embarked"] = df_clean["Embarked"].fillna(mode_embarked)

# 5. Check erroneous values
print("Negative age:", (df_clean["Age"] < 0).sum())
print("Negative fare:", (df_clean["Fare"] < 0).sum())
print("Invalid survival:", (~df_clean["Survived"].isin([0, 1])).sum())
print("Invalid class:", (~df_clean["Pclass"].isin([1, 2, 3])).sum())
print("Invalid embarked:",
      (~df_clean["Embarked"].isin(["C", "Q", "S"])).sum())

# 6. Detect Fare outliers using IQR
Q1 = df_clean["Fare"].quantile(0.25)
Q3 = df_clean["Fare"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df_clean[
    (df_clean["Fare"] < lower) |
    (df_clean["Fare"] > upper)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower:", lower)
print("Upper:", upper)
print("Number of Fare outliers:", len(outliers))

# 7. Visualize before capping
sns.boxplot(x=df_clean["Fare"])
plt.title("Fare Before Outlier Treatment")
plt.show()

# 8. Cap outliers
df_clean["Fare"] = df_clean["Fare"].clip(
    lower=lower,
    upper=upper
)

# 9. Visualize after capping
sns.boxplot(x=df_clean["Fare"])
plt.title("Fare After Outlier Treatment")
plt.show()

# 10. Basic preprocessing
df_clean.drop(
    ["PassengerId", "Name", "Ticket"],
    axis=1,
    inplace=True
)

df_clean = pd.get_dummies(
    df_clean,
    columns=["Sex", "Embarked"],
    drop_first=True
)

# 11. Final validation
print("Final missing values:")
print(df_clean.isnull().sum())

print("Final shape:", df_clean.shape)
print("Duplicate rows:", df_clean.duplicated().sum())

# 12. Save cleaned data
df_clean.to_csv("titanic_cleaned.csv", index=False)

print("Cleaning completed successfully.")