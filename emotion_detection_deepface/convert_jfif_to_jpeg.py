# Convert JFIF in a folder to JPEG images and move to face_extracts folder
# This file will have 1 functions: convert_jfif_to_jpeg. This function will take in a path to a folder and convert all
# JFIF images to JPEG images and move them to the face_extracts folder. This function will return the number of images
# converted.

import os

from PIL import Image


# This function converts JFIF images to JPEG images and moves them to the face_extracts folder. It returns the number
# of images converted.
def convert_jfif_to_jpeg(images_dir):
    # Create face_extracts directory if it doesn't exist
    face_extracted_directory = os.path.join(images_dir, "face_extracts")
    os.makedirs(face_extracted_directory, exist_ok=True)

    # Find .jfif images in the specified directory
    image_paths = [os.path.join(images_dir, filename) for filename in os.listdir(images_dir) if
                   filename.endswith(".jfif")]

    # Convert and move images
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                base_name = os.path.basename(image_path)
                file_name, _ = os.path.splitext(base_name)
                new_file_name = f"{file_name}.jpeg"
                output_path = os.path.join(face_extracted_directory, new_file_name)
                img.convert('RGB').save(output_path, "JPEG")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return len(image_paths)


# Get the current working directory
cwd = os.getcwd()
# Specify the folder containing the images
images_directory = os.path.join(cwd, "daon_faces")  # Replace with the path to your folder

print(f"Converting JFIF images in {images_directory} to JPEG images...")
# Convert images
converted_images_count = convert_jfif_to_jpeg(images_directory)
print(f"Converted {converted_images_count} images.")
