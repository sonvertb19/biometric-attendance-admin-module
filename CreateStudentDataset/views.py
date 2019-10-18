from rest_framework import response
from rest_framework.decorators import api_view
import requests as rq
import json
from FaceRecognition.Create_dataset import run_script
from FaceRecognition.Train import train_faces

import os
import sys


@api_view(["POST"])
def create_student_dataset(request):

    # Making a request to the BasDB_API server.
    #   Adding Custom Auth Header.
    custom_header = {'Authorization': request.headers['Authorization']}

    roll_number = request.data['user']['username']
    print(roll_number)

    rv = run_script(roll_number=roll_number)

    if rv == -1:
        return response.Response(status=500, data={"message": "Face could not be captured.",
                                                   "troubleshoot": "Try again"})

    r = rq.post("http://127.0.0.1:8001/students/", json=request.data, headers=custom_header)

    print("Request with the above data made to the server.")
    print("Response: ", end=" ")
    print(r.text)
    print("status_code: ", end=" ")
    print(r.status_code)

    if r.status_code != 201:
        # Delete all files with id: roll_number
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        facerecog_dir = os.path.join(base_dir, "FaceRecognition")
        dataset_path = os.path.join(facerecog_dir, "Dataset")

        files = os.listdir(dataset_path)

        for file in files:
            try:
                print("Deleting Dataset File: " + str(file))
                if str(roll_number) in file:
                    file_abs_path = os.path.join(dataset_path, file)
                    print(file.split('.')[2])
                    os.remove(file_abs_path)
            except Exception:
                print(sys.exc_info()[0])
                return response.Response(status=500, data=json.loads(sys.exc_info()[0]))

    return response.Response(status=r.status_code, data=json.loads(r.text))


@api_view(["GET"])
def train_model(request):
    ret_val = train_faces()

    if ret_val == -1:
        print("Dataset is empty")
        data = {"error": "Dataset directory is empty. ",
                "troubleshooting": "Add some students before trying to training the model"}

        return response.Response(status=500, data=data)

    return response.Response(status=200, data={"success": "Model trained successfully",
                                               "faces_trained": ret_val})
