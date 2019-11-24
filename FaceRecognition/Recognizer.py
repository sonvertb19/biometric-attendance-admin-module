"""
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
    ==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
    ==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

"""


import cv2
import numpy as np
import os 

def recognizer():
    identified_students = set()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FACERECOG_DIR = os.path.join(BASE_DIR, "FaceRecognition")
    CCPATH = os.path.join(FACERECOG_DIR, "Cascades", "haarcascade_frontalface_default.xml")
    DATASET_PATH = os.path.join(FACERECOG_DIR, "Dataset")
    TRAINER_PATH = os.path.join(FACERECOG_DIR, "trainer", "trainer.yml")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    faceCascade = cv2.CascadeClassifier(CCPATH);

    font = cv2.FONT_HERSHEY_SIMPLEX

    # initiate id counter
    id = 0

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    loop_counter = 4000

    # Will capture 4000 frames and exit.
    while loop_counter:
        loop_counter = loop_counter - 1

        ret, img = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
           )

        for(x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # print("confidence: " + str(confidence) + " " + " id: " + str(id))

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if confidence < 60:
                confidence = "  {0}%".format(round(100 - confidence))

                identified_students.add(id)
                # if data[str(id)] == 0:
                #     data[str(id)] = 1
                #     print("Attendance marked for " + (str(id)))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break



    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

    return identified_students


if __name__ == "__main__":
    print(recognizer())