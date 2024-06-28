import torch
from ultralytics import YOLO

# Define the paths to the models and datasets
models = [
    "runs/classify/train45/weights/last.pt",
    "runs/classify/train46/weights/last.pt",
    "runs/classify/train47/weights/last.pt",
    "runs/classify/train48/weights/last.pt",
    "runs/classify/train49/weights/last.pt",
]

datasets = [
    "fold_reduced_1",
    "fold_reduced_2",
    "fold_reduced_3",
    "fold_reduced_4",
    "fold_reduced_5",
]

"""datasets = [
    "Nils",
    "Lea",
    "Max",
    "Benjamin",
    "Max"
]"""

scores = []

# Iterate over each model and dataset pair
for model_path, dataset_name in zip(models, datasets):
    # Load the model using YOLO
    model = YOLO(model_path)

    # Load the dataset
    # Assuming you have a function to load the dataset
    # Validate the model
    results = model.val(data=dataset_name)
    # Print the results
    print(f"Model: {model_path}, Dataset: {dataset_name}, Results: {results}")

# Optionally, you can save the scores to a file or visualize them
