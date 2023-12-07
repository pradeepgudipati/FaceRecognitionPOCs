import json
import os
import random

import numpy as np
from PIL import Image
from deepface import DeepFace
from emotion_detection_fer2013.src.emotions_predictor_fer2013 import predict_emotion

cwd = os.getcwd()
images_directory = os.path.join(cwd, "images")
face_extracted_directory = os.path.join(cwd, "face_extracts")

backends = [
    'opencv',  # 0
    'ssd',  # 1
    'dlib',  # 2
    'mtcnn',  # 3
    'retinaface',  # 4
    'mediapipe',  # 5
    'yolov8',  # 6
    'yunet',  # 7
    'fastmtcnn',  # 8
]


def reset_dirs():
    # Create directories if they do not exist
    os.makedirs(images_directory, exist_ok=True)
    os.makedirs(face_extracted_directory, exist_ok=True)
    # Delete all files in the face extracted directory
    for file in os.listdir(face_extracted_directory):
        os.remove(os.path.join(face_extracted_directory, file))


def load_and_extract_all_faces(imgs_dir):
    image_paths = [os.path.join(imgs_dir, filename) for filename in os.listdir(imgs_dir) if
                   filename.endswith((".jpeg", ".png", ".jpg"))]
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        load_and_extract_face(image_path)


def save_extracted_face(face_data, base_name):
    face_array = face_data['face']
    if isinstance(face_array, np.ndarray):
        face_image = Image.fromarray((face_array * 255).astype('uint8'))
    else:
        raise TypeError("The face object is not a numpy array.")

    file_name, file_extension = os.path.splitext(base_name)
    new_file_name = f"{file_name}_face_{random.randint(100, 999)}{file_extension}"
    output_path = os.path.join(face_extracted_directory, new_file_name)
    face_image.save(output_path)
    return output_path, face_image


def load_and_extract_face(image_path):
    base_name = os.path.basename(image_path)
    face_objects = DeepFace.extract_faces(image_path, target_size=(224, 224), detector_backend=backends[3],
                                          enforce_detection=True)
    if not face_objects:
        print(f"No faces detected in image {image_path}")
        return
    for face_data in face_objects:
        extracted_face_path, _ = save_extracted_face(face_data, base_name)
        analyze_face(extracted_face_path)


def load_and_analyze_all_faces(images_dir_name, start=0, end=0):
    print(images_dir_name)
    images_path = os.path.join(cwd, images_dir_name)
    filename = 'image_analysis_' + images_dir_name + ".html"
    print(f"Writing HTML file to {filename}")

    image_analysis_pairs = []

    image_paths = [os.path.join(images_path, filename) for filename in os.listdir(images_path) if
                   filename.endswith((".jpeg", ".png", ".jpg"))]
    if start == 0 and end == 0:
        start = 0
        end = len(image_paths)

    count = 0
    for image_path in image_paths:
        if count > start:
            analysis = analyze_face(image_path, count)
            emotion = predict_emotion(image_path)
            if analysis:
                image_analysis_pairs.append((image_path, analysis, emotion))

        count += 1
        if count == end:
            break  # For testing purposes, only analyze the first 25 images

    html_content = create_html_content(image_analysis_pairs)

    with open(filename, 'w') as file:
        file.write(html_content)
        print(f"HTML file written to {file.name}")

    return html_content


# Analyze a face in an image using the DeepFace library analyze function
def analyze_face(face_img_path, count=0):
    print(f"Analyzing face in image {count} : {face_img_path}")
    try:
        analyses = DeepFace.analyze(face_img_path, detector_backend=backends[3],
                                    actions=['age', 'gender', 'race', 'emotion'])

        # Assuming we want to analyze only the first face detected
        first_analysis = analyses[0] if isinstance(analyses, list) else analyses

        return first_analysis
    except Exception as e:
        print(f"Error analyzing face in image {face_img_path}: {e}")
        return None


def create_html_content(image_analysis_pairs):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Analysis</title>
        <style>
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid black; padding: 10px; text-align: left; }
            .json-cell { font-family: monospace; white-space: pre; }
        </style>
    </head>
    <body>
        <h2>Image Analysis Results</h2>
        <table>
            <tr>
                <th>Image</th>
                <th>Analysis JSON</th>
            </tr>"""

    count = 0
    for image_path, analysis, emotion in image_analysis_pairs:
        analysis_str = json.dumps(analysis, indent=4)  # Convert analysis dict to a pretty-printed string
        html_content += f"""
            <tr>
                <td>
                    Image -- {count} - {image_path}<br>
                    <img src="{image_path}" alt="Analyzed Image" style="width: 300px; height: 300px;">
                </td>
                <td class="json-cell">{analysis_str}</td>
                <td class="json-cell">Fer2013 based -- Emotion  = {emotion}</td>
            </tr>"""
        count += 1

    html_content += """
        </table>
    </body>
    </html>"""

    return html_content


# load_and_extract_all_faces(images_directory)
load_and_analyze_all_faces("../Data/Selfies")
