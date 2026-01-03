import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ComplaintEDA:
    def __init__(self, df):
        """
        Load the full CFPB dataset.
        """
        self.df = df
        print("Dataset loaded successfully.")
        print("Shape:", self.df.shape)

    # ---------------------------
    # 1. Initial EDA
    # ---------------------------
    def basic_overview(self):
        print("\n--- BASIC INFO ---")
        print(self.df.info())
        print("\n--- SUMMARY ---")
        print(self.df.describe(include="all"))

    # ---------------------------
    # 2. Distribution of complaints by product
    # ---------------------------
    def product_distribution(self):
        print("\n--- PRODUCT DISTRIBUTION ---")
        print(self.df["Product"].value_counts())

        plt.figure(figsize=(10, 5))
        self.df["Product"].value_counts().plot(kind="bar", color="skyblue")
        plt.title("Distribution of Complaints by Product")
        plt.xlabel("Product")
        plt.ylabel("Count")
        plt.show()

    # ---------------------------
    # 3. Narrative length distribution
    # ---------------------------
    def narrative_length(self):
        print("\n--- COMPUTING NARRATIVE LENGTH ---")

        self.df["narr_len"] = (
            self.df["Consumer complaint narrative"]
            .astype(str)
            .apply(lambda x: len(x.split()))
        )

        print(self.df["narr_len"].describe())

        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["narr_len"], bins=50)
        plt.title("Distribution of Narrative Length (Word Count)")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.show()

    # ---------------------------
    # 4. Complaints with vs without narratives
    # ---------------------------
    def narrative_presence(self):
        print("\n--- NARRATIVE PRESENCE ---")

        with_narr = self.df["Consumer complaint narrative"].notna().sum()
        without_narr = self.df["Consumer complaint narrative"].isna().sum()

        print("Complaints WITH narrative:", with_narr)
        print("Complaints WITHOUT narrative:", without_narr)

        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=["With Narrative", "Without Narrative"],
            y=[with_narr, without_narr],
            palette="Blues"
        )
        plt.title("Complaints With vs Without Narratives")
        plt.show()

    # ---------------------------
    # Run all EDA steps at once
    # ---------------------------
    def run_all(self):
        self.basic_overview()
        self.product_distribution()
        self.narrative_length()
        self.narrative_presence()
