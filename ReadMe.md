# Face Emotion Recognition from Images 


# This Project has 2 sub projects using 2 different models to detect emotions from images.

## DeepFace
The Code under ```\emotion_detection_deepface``` uses the DeepFace library to detect emotions from images. The code is based on the [DeepFace Library](https://github.com/serengil/deepface)
## FER 2013 based model 
The Code under ```\emotion_detection_fer2013``` uses the FER 2013 dataset to train a model to detect emotions from images. The code is based on the [FER 2013 dataset](https://www.kaggle.com/deadskull7/fer2013)
The pre-trained model is saved in the ```\emotion_detection_fer2013\src``` folder. The model is trained for 50 epochs and has an accuracy of 63.2% on the test data.

### To run the code, follow the steps below:
#### 1. Copy all your images to a folder 
#### 2. Go to ```face_emotion_detection.py``` file, 
#### 3. Change the last line that calls ```load_and_analyze_all_faces(<images path>)``` 
#### 4. Replace the ```<images path>``` with the full path of where the images are located
#### 5. Run the code and wait for the results
#### 6. The results will be saved in an HTML file in the same folder as the images
#### 7. The HTML file will have the image links and the Deepface analysis JSON + emotion detected from each image in 3 columns
#### 8. The load_and_analyze_all_faces function calls the ```predict_emotion(image_path)``` method in ```emotion_predictor_fer2013.py``` file which is the main file that does the emotion detection based on fer2013 model

