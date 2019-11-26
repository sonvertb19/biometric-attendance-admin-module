"""
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
    ==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
    ==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

"""


import cv2
import numpy as np
from PIL import Image
import os


# function to get the images and label data
def getImagesAndLabels(path):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FACERECOG_DIR = os.path.join(BASE_DIR, "FaceRecognition")
    CCPATH = os.path.join(FACERECOG_DIR, "Cascades", "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(CCPATH);
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        # convert it to gray scale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids


def train_faces():
    # Path for face image database
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FACERECOG_DIR = os.path.join(BASE_DIR, "FaceRecognition")
    DATASET_PATH = os.path.join(FACERECOG_DIR, "Dataset")
    TRAINER_FOLDER_PATH = os.path.join(FACERECOG_DIR, "trainer") 
    TRAINER_PATH = os.path.join(FACERECOG_DIR, "trainer", "trainer.yml")

    if not os.path.exists(TRAINER_FOLDER_PATH):
            os.makedirs(TRAINER_FOLDER_PATH)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(DATASET_PATH)

    if len(ids) == 0:
        print("\n[Error] Dataset is empty!!")
        return -1

    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write(TRAINER_PATH)

    # Print the number of faces trained and end program
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    faces_trained = len(np.unique(ids))
    return faces_trained


if __name__ == '__main__':
    retVal = train_faces()
    if retVal == -1:
        print("An error terminated the train operation!!")
