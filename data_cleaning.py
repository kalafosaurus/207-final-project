from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path

from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore


def load_metadata(metadata_path: Path, image_path: Path) -> pd.DataFrame:
    """Load the metadata for each data split."""
    # Load the metadata for each split
    md_train = pd.read_csv(metadata_path / "FungiTastic-FewShot-Train.csv")
    md_val = pd.read_csv(metadata_path / "FungiTastic-FewShot-Val.csv")
    md_test = pd.read_csv(metadata_path / "FungiTastic-FewShot-Test.csv")
    
    # Label each split
    md_train["split"] = "train"
    md_val["split"] = "val"
    md_test["split"] = "test"

    # Join all of the data together
    full_df = pd.concat([md_train, md_val, md_test])

    # Add the full image location for each image
    # Options for image size include 300p, 500p, 720p, fullsize
    full_df["image_path"] = full_df.apply(
        lambda row: image_path / f"{row['split']}/300p/{row['filename']}", axis=1
    )

    return full_df


def filter_low_counts(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Filter out examples of fungi with low value counts."""
    class_counts = df["class"].value_counts()
    frequent_classes = class_counts[class_counts >= min_samples].index
    filtered_df = df[df["class"].isin(frequent_classes)]
    return filtered_df


def load_images_and_labels(df: pd.DataFrame, image_size: Optional[tuple[int, int]]):
    """Load the images and labels based on the metadata frame passed in."""
    images = []
    labels = []

    for idx, row in df.iterrows():
        # Load and save the image as an array
        img = load_img(row["image_path"], target_size=image_size)
        img_arr = img_to_array(img)
        images.append(img_arr)

        # Append the class to the list of labels
        labels.append(row["class_idx"])

    # Stack and convert into a numpy array
    images = np.stack(images)

    # Rescale all of the images so they're pixel value 0 - 1
    images = images / 255.0

    # Cast label list to np.array for easier manipulation
    labels = np.array(labels)

    return images, labels
