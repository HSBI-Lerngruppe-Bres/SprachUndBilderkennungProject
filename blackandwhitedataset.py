import os
import cv2


def convert_and_resize_images(input_folder, output_folder, size=(28, 28)):
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
                # Resize the image
                resized_image = cv2.resize(gray_image, size)
                # Construct the output file path
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                # Save the resized grayscale image
                cv2.imwrite(os.path.join(output_path, file), resized_image)


# Path to your CIFAR-100 dataset folder
input_folder = 'datasets/Max/train'
# Path to save the black and white images
output_folder = 'datasets/MaxE/val'

convert_and_resize_images(input_folder, output_folder)
