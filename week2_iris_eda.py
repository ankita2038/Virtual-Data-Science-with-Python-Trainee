# Week 2: Exploratory Data Analysis and Visualization
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df.drop(columns=["target"], inplace=True)

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df["species"].value_counts())

summary = df.groupby("species")[
    ["sepal length (cm)", "sepal width (cm)",
     "petal length (cm)", "petal width (cm)"]
].mean()
print(summary)

corr = df.drop(columns="species").corr()
print(corr)

# Visualizations
sns.countplot(data=df, x="species")
plt.title("Iris Species Distribution")
plt.xlabel("Species"); plt.ylabel("Number of Samples")
plt.show()

sns.histplot(data=df, x="sepal length (cm)", hue="species", kde=True)
plt.title("Sepal Length Distribution by Species")
plt.xlabel("Sepal Length (cm)"); plt.ylabel("Number of Samples")
plt.show()

sns.boxplot(data=df, x="species", y="petal length (cm)")
plt.title("Petal Length by Species")
plt.xlabel("Species"); plt.ylabel("Petal Length (cm)")
plt.show()

sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species")
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)"); plt.ylabel("Sepal Width (cm)")
plt.show()

sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="species")
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)"); plt.ylabel("Petal Width (cm)")
plt.show()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

for col in iris.feature_names:
    Q1=df[col].quantile(.25); Q3=df[col].quantile(.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR; upper=Q3+1.5*IQR
    outliers=df[(df[col]<lower)|(df[col]>upper)]
    print(col, "potential outliers:", len(outliers))

sns.boxplot(data=df.drop(columns="species"))
plt.title("Numerical Feature Distributions")
plt.xlabel("Features"); plt.ylabel("Measurement (cm)")
plt.show()

df.to_csv("iris_week2_dataset.csv", index=False)
print("Week 2 EDA completed.")
