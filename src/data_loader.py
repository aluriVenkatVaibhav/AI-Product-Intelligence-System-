import pandas as pd
from pathlib import Path

from src.config import STYLE_CSV, IMAGE_DIR

class FashionDataset:

    def __init__(self):
        self.df = None

    def load(self):
        """
        Load styles.csv
        """
        self.df = pd.read_csv(
            STYLE_CSV,
            on_bad_lines="skip"
        )

        return self.df

    def clean(self):
        """
        Remove rows with missing data
        """
        if self.df is None:
            self.load()

        columns = [
            "id",
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "usage",
            "productDisplayName",
        ]

        self.df = self.df[columns]

        self.df.dropna(inplace=True)

        return self.df

    def remove_missing_images(self):
        """
        Remove rows whose image doesn't exist
        """

        if self.df is None:
            self.clean()

        valid_rows = []

        for _, row in self.df.iterrows():

            image_path = IMAGE_DIR / f"{row['id']}.jpg"

            if image_path.exists():
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows)

        return self.df

    def get_dataframe(self):

        if self.df is None:
            self.load()
            self.clean()
            self.remove_missing_images()

        return self.df
    
    def save_metadata(self, output_path):
        """
        Save cleaned metadata for later use.
        """
        self.df.to_parquet(output_path, index=False)