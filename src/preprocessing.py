import re
import os
import pandas as pd

class ComplaintPreprocessor:
    def __init__(self, df, save_path=r"E:\Intelligent-Complaint-Analysis-for-Financial-Services-week_7\data\processed\filtered_complaints.csv"):
        self.df = df.copy()
        self.save_path = save_path

        # Allowed product list
        self.allowed_products = [
            "Credit card",
            "Personal loan",
            "Savings account",
            "Money transfers"
        ]

    def filter_products(self):
        """Keep only the required products."""
        self.df = self.df[self.df["Product"].isin(self.allowed_products)]
        return self

    def remove_empty_narratives(self):
        """Remove records with missing or blank complaint narratives."""
        self.df = self.df[self.df["Consumer complaint narrative"].notna()]
        self.df = self.df[self.df["Consumer complaint narrative"].str.strip() != ""]
        return self

    def clean_text(self, text):
        """Clean narrative for embedding."""
        
        text = text.lower()  # Lowercase
        
        # Remove boilerplate patterns
        boilerplate_patterns = [
            r"i am writing to file a complaint",
            r"this is a complaint",
            r"to whom it may concern",
            r"i want to report",
        ]
        for p in boilerplate_patterns:
            text = re.sub(p, "", text)

        # Remove non-letters except basic punctuation
        text = re.sub(r"[^a-z0-9\s\.,!?]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def apply_cleaning(self):
        """Apply text cleaning to all narratives."""
        self.df["cleaned_narrative"] = self.df["Consumer complaint narrative"].apply(self.clean_text)
        return self

    def save_output(self):
        """Save cleaned dataset to disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.df.to_csv(self.save_path, index=False)
        print(f"✔ Cleaned & filtered dataset saved to: {self.save_path}")
        return self

    def run_all(self):
        """Full preprocessing pipeline."""
        (
            self.filter_products()
                .remove_empty_narratives()
                .apply_cleaning()
                .save_output()
        )
        return self.df
