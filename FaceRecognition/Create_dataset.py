"""
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
    ==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
    ==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    
"""

import cv2
import os
import numpy as np
import imutils
import math


def run_script(roll_number):
    try:
        face_id = roll_number
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        FACERECOG_DIR = os.path.join(BASE_DIR, "FaceRecognition")

        CCPATH = os.path.join(FACERECOG_DIR, "Cascades", "haarcascade_frontalface_default.xml")
        DATASET_PATH = os.path.join(FACERECOG_DIR, "Dataset")
        #Making the dataset directory if it doesn't already exist.
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        print(CCPATH)
        face_detector = cv2.CascadeClassifier(CCPATH)

        # For each person, enter one numeric face id
        # face_id = "204"

        print("\n [INFO] Initializing face capture. Look in the camera and wait ...")
        # Initialize individual sampling face count
        count = 0

        while True:

            ret, img = cam.read()
            # img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # Save the captured image into the datasets folder
                path_for_image = os.path.join(DATASET_PATH, "User." + str(face_id) + '.' + str(count) + ".jpg")
                print(path_for_image)
                print("Status: " + str(cv2.imwrite(path_for_image, gray[y:y + h, x:x + w])))

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 60:  # Take 60 face sample and stop video
                break

        cam.release()

        files = os.listdir(DATASET_PATH)

        for file in files:
            # print(files)
            # print("in loop")
            full_name = os.path.join(DATASET_PATH, file)
            # print(full_name)
            img = cv2.imread(full_name)
            id = int(os.path.split(full_name)[-1].split(".")[1])
            # print(id)
            if int(id) == int(face_id):
                # adding salt and pepper noise, mean and sigma can be altered
                im = np.zeros(img.shape, np.uint8)  # do not use original image it overwrites the image
                mean = 15
                sigma = 5
                cv2.randn(im, mean, sigma)  # create the random distribution
                Noise = cv2.add(img, im)  # add the noise to the original image

                # Writing the file after processing
                count += 1

                path_for_image = os.path.join(DATASET_PATH, "User." + str(face_id) + '.' + str(count) + ".jpg")
                # print(path_for_image)

                print("Status: " + str(cv2.imwrite(path_for_image, img)))

                # adding rotation

                roted_image = rotate(img, 15)

                # Writing the file after processing
                count += 1
                path_for_image = os.path.join(DATASET_PATH, "User." + str(face_id) + '.' + str(count) + ".jpg")
                # print(path_for_image)

                print("Status: " + str(cv2.imwrite(path_for_image, roted_image)))
                # cv2.imwrite(path1,Noise)

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cv2.destroyAllWindows()
    except:
        return -1    

def rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    angle = math.radians(angle)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)


def crop(img, w, h):
    x, y = int(img.shape[1] * .5), int(img.shape[0] * .5)

    return img[
           int(np.ceil(y - h * .5)): int(np.floor(y + h * .5)),
           int(np.ceil(x - w * .5)): int(np.floor(x + h * .5))
           ]


def rotate(img, angle):
    # rotate, crop and return original siz
    (h, w) = img.shape[:2]
    img = imutils.rotate_bound(img, angle)
    img = crop(img, *rotated_rect(w, h, angle))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


if __name__ == '__main__':

    face_id = input('\n enter user id end press <return> ==>  ')
    run_script(face_id)
