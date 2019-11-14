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


@api_view(["POST"])
def mark_attendance(request):
    """
    1. Get timetable_period (id) and room_number (name) from request data.
    2. Request the details of the timetable.    (Asynchroniously)
    3. Get the section (id) from the timetable.      (Asynchroniously)
    4. list of students of the section.         (Asynchroniously)

    """

    ##########################################################
    # IN ORDER TO SELECT CO1, CO2, CO3.
    # HOW WILL THE TEACHER TELL THAT THERE ARE THREE SECTIONS
    # BY SELECTING THREE TIMETABLES?
    # BECAUSE TIMETABLE IS FOR A **SECTION** AND NOT **MAIN_SECTION**
    ##########################################################

    data = request.data
    timetable_period_id = data["timetable_period"]
    print(data)
    print(timetable_period_id)

    custom_header = {"Authorization": request.headers["Authorization"]}

    # TODO: Make it async
    r = rq.get("http://127.0.0.1:8001/timetable_periods/{0}/".format(timetable_period_id),
                headers=custom_header)
    if r.status_code != 200:
        return response.Response(status = r.status_code, data=json.loads(r.text))
    print(json.dumps(json.loads(r.text), indent=4))

    # Parse response as json and get section id from it.
    timetable_object = json.loads(r.text)

    section_id = timetable_object["subject"]["section"]
    print("section id: {0}".format(section_id))

    # Get list of all students in the section(s)
    return response.Response(status=200, data={"success": "fakeSuccess"})