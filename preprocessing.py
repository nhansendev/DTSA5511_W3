from tifffile import imread
import os
import numpy as np
import pandas as pd
from functools import partial
from tqdm.contrib.concurrent import process_map


SCRIPT_DIR = os.getcwd()


def get_image(dir, ID):
    # Load tif image as a numpy array
    return imread(os.path.join(dir, ID + ".tif"))


def generate_numpy_file(
    labels, folder, fname="train", split=1, max_workers=20, chunksize=1024
):
    # Convert separate image files into saved numpy arrays,
    # which are faster to read from disk

    # Several sub-arrays can be created as separate files by specifying the "split"
    # Useful for machines with limited RAM

    # Adjust max_workers to something reasonable for your CPU

    # For multiprocessing
    _read = partial(get_image, folder)

    hasLabels = "label" in labels.columns

    IDs = labels["id"].values
    if hasLabels:
        label_vals = labels["label"].values

    # Interval between indexes for splitting between files
    ival = int(np.ceil(len(IDs) / split))

    # For each output file
    for i in range(split):
        print(f"Reading Data {i+1} of {split}")
        id_subset = IDs[ival * i : ival * (i + 1)]
        if hasLabels:
            label_subset = label_vals[ival * i : ival * (i + 1)]

        print(f"{len(id_subset)} images")
        images = process_map(
            _read, id_subset, max_workers=max_workers, chunksize=chunksize
        )
        # Save on memory while converting to 0-1 range float instead of integer
        images = np.array(images).astype(np.float16) / 256
        # Pytorch expects a shape like: [batch_size, channels, W, H]
        images = np.reshape(images, (-1, 3, 96, 96))

        print(f"Writing Data {i+1}")
        np.save(os.path.join(SCRIPT_DIR, "Datasets", f"{fname}_{i}"), images)
        if hasLabels:
            np.save(
                os.path.join(SCRIPT_DIR, "Datasets", f"{fname}_labels_{i}"),
                label_subset,
            )


def get_numpy_file_names(folder, basename="train"):
    # Get the saved numpy array file names for data and labels
    tmp = [
        f
        for f in os.listdir(folder)
        if f.startswith(basename) and f.endswith(".npy") and "_" in f
    ]
    return [t for t in tmp if "labels" not in t], [t for t in tmp if "labels" in t]


if __name__ == "__main__":
    # train_labels = pd.read_csv(os.path.join(SCRIPT_DIR, "Datasets", "train_labels.csv"))
    # train_folder = os.path.join(SCRIPT_DIR, "Datasets", "train")

    # generate_numpy_file(train_labels, train_folder, "train")

    test_labels = pd.read_csv(os.path.join(SCRIPT_DIR, "Datasets", "test_labels.csv"))
    test_folder = os.path.join(SCRIPT_DIR, "Datasets", "test")

    generate_numpy_file(test_labels, test_folder, "test")
