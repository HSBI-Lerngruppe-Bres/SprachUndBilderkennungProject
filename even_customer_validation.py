import itertools
import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load images from a directory


def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = os.listdir(folder)
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert to grayscale if necessary
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(img)
                    labels.append(class_name)
    return images, labels, class_names

# Function to perform validation


def validate_model(folder, output_path, out_file_name):
    images, true_labels, class_names = load_images_from_folder(folder)
    class_names = sorted(class_names)
    y_true = []
    y_pred = []
    confidences = []

    for img, true_label in zip(images, true_labels):
        results = model(img)
        for result in results:
            probs = result.probs.cpu().numpy()
            class_idx = np.argmax(probs.data)
            predicted_label = class_names[class_idx]
            confidence = probs.data[class_idx]

            y_true.append(true_label)
            y_pred.append(predicted_label)
            confidences.append(confidence)

    # Calculate metrics
    y_true_idx = [class_names.index(label) for label in y_true]
    y_pred_idx = [class_names.index(label) for label in y_pred]

    avg_confidence = np.mean(confidences)
    precision = precision_score(y_true_idx, y_pred_idx, average='weighted')
    recall = recall_score(y_true_idx, y_pred_idx, average='weighted')
    f1 = f1_score(y_true_idx, y_pred_idx, average='weighted')
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    conf_matrix = confusion_matrix(y_true_idx, y_pred_idx)

    # Transpose the confusion matrix
    conf_matrix = conf_matrix.T

    print(f'Average Confidence: {avg_confidence:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("path created")
    print("1")

    # Plot transposed confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, out_file_name + ".png"))
    plt.close()
    print("2")

    with open(os.path.join(output_path, out_file_name + ".txt"), "w") as file:
        file.write(f'Average Confidence: {avg_confidence:.2f}\n')
        file.write(f'Precision: {precision:.2f}\n')
        file.write(f'Recall: {recall:.2f}\n')
        file.write(f'F1 Score: {f1:.2f}\n')
        file.write(f'Accuracy: {accuracy:.2f}\n')
        file.write('Confusion Matrix:\n')
        file.write(str(conf_matrix))
        file.close()
        print("3")


# Specify the folder containing class subfolders with images
# Load your trained YOLO classification model
models = ['runs/classify/train193/weights/best.pt',
          'runs/classify/train194/weights/best.pt',
          'runs/classify/train195/weights/best.pt',
          'runs/classify/train196/weights/best.pt',
          'runs/classify/train197/weights/best.pt']

data_folders = ['datasets/fold_1/val',
                'datasets/fold_2/val',
                'datasets/fold_3/val',
                'datasets/fold_4/val',
                'datasets/fold_5/val']

file_names = ["fold_1",
              "fold_2",
              "fold_3",
              "fold_4",
              "fold_5"]

outfolder = "ecv/normal_val"
"""for modelstr, data_folder, file_name in zip(models, data_folders, file_names):
    model = YOLO(modelstr)
    validate_model(data_folder, outfolder, file_name)"""

outfolder = "ecv/own_data_val"

data_folders = [('datasets/Benjamin/val', "benji"),
                ('datasets/Lea/val', "lea"),
                ('datasets/Max/val', "max"),
                ('datasets/Nils/val', "nils"),]

models = [('runs/classify/train193/weights/best.pt', "fold_1"),
          ('runs/classify/train194/weights/best.pt', "fold_2"),
          ('runs/classify/train195/weights/best.pt', "fold_3"),
          ('runs/classify/train196/weights/best.pt', "fold_4"),
          ('runs/classify/train197/weights/best.pt', "fold_5")]

for (data_folder, name1), (modelstr, name2) in set(itertools.product(data_folders, models)):
    model = YOLO(modelstr)
    validate_model(data_folder, outfolder, f"{name1}_{name2}")
