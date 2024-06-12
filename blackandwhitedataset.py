import os
from PIL import Image
import cv2


def convert_to_black_and_white(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                # Construct full file path
                file_path = os.path.join(root, file)
                # Read the image
                image = cv2.imread(file_path)
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Construct the output file path
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                # Save the grayscale image
                cv2.imwrite(os.path.join(output_path, file), gray_image)


# Path to your CIFAR-100 dataset folder
input_folder = 'path/to/your/cifar100/folder'
# Path to save the black and white images
output_folder = 'path/to/save/black_and_white_images'

convert_to_black_and_white(input_folder, output_folder)
