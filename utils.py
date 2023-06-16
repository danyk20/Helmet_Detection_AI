import os
import shutil
from collections import defaultdict
from PIL import Image

def copy_directory_with_unique_files(directory_path, copy_directory_path):
    # Create the copy directory if it doesn't exist
    if not os.path.exists(copy_directory_path):
        os.makedirs(copy_directory_path)

    # Create a hashmap to store file sizes as keys and file names as values
    file_sizes = defaultdict(list)

    # Iterate over all files in the original directory
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            size = os.path.getsize(file_path)
            file_sizes[size].append(filename)

    # Iterate over the hashmap and copy files with unique sizes to the copy directory
    for size, filenames in file_sizes.items():
        if len(filenames) == 1:
            filename = filenames[0]
            src_file = os.path.join(directory_path, filename)
            dest_file = os.path.join(copy_directory_path, filename)
            shutil.copy2(src_file, dest_file)

    print("Copy operation completed successfully!")

def resize_images(input_folder, output_folder, target_size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Open the image file
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Resize the image while maintaining the aspect ratio
            image.thumbnail(target_size, Image.ANTIALIAS)

            # Create a new image with white background
            new_image = Image.new("RGB", target_size, (255, 255, 255))

            # Calculate the position to paste the resized image
            position = ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2)

            # Paste the resized image onto the new image
            new_image.paste(image, position)

            # Save the new image
            output_path = os.path.join(output_folder, filename)
            new_image.save(output_path)

            print(f"Resized and saved {filename}")

# Example usage:
original_directory = "/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/helmet"
copy_directory = "/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/helmet_a"
# copy_directory_with_unique_files(original_directory, copy_directory)

input_folder = "/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/helmet_a"
output_folder = "/Users/danielkosc/Documents/MUNI/Spring2023/ML/project/data_copy/helmet_b"
target_size = (128, 128)

# Resize the images
resize_images(input_folder, output_folder, target_size)