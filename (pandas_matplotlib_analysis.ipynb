# Analyzing Data with Pandas and Visualizing Results with Matplotlib + Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Apply seaborn style
sns.set(style="whitegrid")

def main():
    try:
        # ----------------------------
        # Task 1: Load and Explore Data
        # ----------------------------
        print("Loading Iris dataset...")

        iris = load_iris(as_frame=True)
        df = iris.frame  # Convert to Pandas DataFrame

        print("\nFirst few rows:")
        print(df.head())

        print("\nInfo about dataset:")
        print(df.info())

        print("\nMissing values per column:")
        print(df.isnull().sum())

        # ----------------------------
        # Task 2: Basic Data Analysis
        # ----------------------------
        print("\nStatistical summary:")
        print(df.describe())

        print("\nGroup mean of features by species:")
        print(df.groupby("target").mean())

        # ----------------------------
        # Task 3: Data Visualization
        # ----------------------------

        # Line chart: Sepal Length over index
        plt.figure(figsize=(6,4))
        sns.lineplot(x=df.index, y="sepal length (cm)", data=df, color="blue")
        plt.title("Line Chart of Sepal Lengths")
        plt.xlabel("Index")
        plt.ylabel("Sepal Length (cm)")
        plt.show()

        # Bar chart: Average Petal Length per Species
        plt.figure(figsize=(6,4))
        sns.barplot(x="target", y="petal length (cm)", data=df, ci=None, palette="muted")
        plt.title("Average Petal Length per Species")
        plt.xlabel("Species (0,1,2)")
        plt.ylabel("Petal Length (cm)")
        plt.show()

        # Histogram: Sepal Width Distribution
        plt.figure(figsize=(6,4))
        sns.histplot(df["sepal width (cm)"], bins=20, kde=True, color="green")
        plt.title("Histogram of Sepal Width")
        plt.xlabel("Sepal Width (cm)")
        plt.ylabel("Frequency")
        plt.show()

        # Scatter plot: Sepal Length vs Sepal Width
        plt.figure(figsize=(6,4))
        sns.scatterplot(
            x="sepal length (cm)",
            y="sepal width (cm)",
            hue="target",
            palette="deep",
            data=df
        )
        plt.title("Scatter Plot: Sepal Length vs Width")
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Sepal Width (cm)")
        plt.legend(title="Species")
        plt.show()

        print("\nAnalysis complete. Plots displayed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
