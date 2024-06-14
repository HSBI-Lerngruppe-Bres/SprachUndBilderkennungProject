import os
import shutil
import random

# Define the paths
base_folder = 'datasets/V2'
current_test_folder = os.path.join(base_folder, 'test')
train_folder = os.path.join(base_folder, 'train')
new_test_folder = os.path.join(base_folder, 'val')

# Create train and new test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(new_test_folder, exist_ok=True)

# List all class folders
class_folders = [d for d in os.listdir(current_test_folder) if os.path.isdir(
    os.path.join(current_test_folder, d))]

for class_folder in class_folders:
    # Create class directories in train and new test folders
    os.makedirs(os.path.join(train_folder, class_folder), exist_ok=True)
    os.makedirs(os.path.join(new_test_folder, class_folder), exist_ok=True)

    # List all files in the current class folder
    class_path = os.path.join(current_test_folder, class_folder)
    all_files = [f for f in os.listdir(
        class_path) if os.path.isfile(os.path.join(class_path, f))]

    # Shuffle the files
    random.shuffle(all_files)

    # Calculate the split index
    split_index = int(len(all_files) * 0.1)

    # Split the files into new test and train sets
    new_test_files = all_files[:split_index]
    train_files = all_files[split_index:]

    # Move files to the respective directories
    for file in new_test_files:
        shutil.move(os.path.join(class_path, file), os.path.join(
            new_test_folder, class_folder, file))

    for file in train_files:
        shutil.move(os.path.join(class_path, file),
                    os.path.join(train_folder, class_folder, file))

    print(f"Moved {len(new_test_files)} files to {
          os.path.join(new_test_folder, class_folder)}")
    print(f"Moved {len(train_files)} files to {
          os.path.join(train_folder, class_folder)}")
