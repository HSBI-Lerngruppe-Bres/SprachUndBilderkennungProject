import os
import shutil
from sklearn.model_selection import KFold

# Define the path to your dataset
original_dataset_dir = 'datasets/V3/all_reduced'
new_dataset_dir = 'datasets'
file_prefix = "fold_reduced_"

# Create the new dataset directory if it doesn't exist
os.makedirs(new_dataset_dir, exist_ok=True)

# Get class names from the original dataset
class_names = os.listdir(original_dataset_dir)

# Create the 5 folders for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Get all file paths for each class
file_paths = {class_name: [os.path.join(original_dataset_dir, class_name, fname)
                           for fname in os.listdir(os.path.join(original_dataset_dir, class_name))]
              for class_name in class_names}

# Create directories for each fold
for fold_idx, (train_index, test_index) in enumerate(kf.split(list(file_paths[class_names[0]]))):
    fold_dir = os.path.join(new_dataset_dir, f'{file_prefix}{fold_idx + 1}')
    train_dir = os.path.join(fold_dir, 'train')
    test_dir = os.path.join(fold_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Get file paths for this class
        files = file_paths[class_name]

        # Split into train and test using indices from KFold
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # Copy train files
        for file_path in train_files:
            shutil.copy(file_path, os.path.join(
                train_dir, class_name, os.path.basename(file_path)))

        # Copy test files
        for file_path in test_files:
            shutil.copy(file_path, os.path.join(
                test_dir, class_name, os.path.basename(file_path)))

print("Datasets for cross-validation created successfully!")
