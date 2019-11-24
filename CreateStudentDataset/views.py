from rest_framework import response
from rest_framework.decorators import api_view
import requests as rq
import json
from FaceRecognition.Create_dataset import run_script
from FaceRecognition.Train import train_faces
from FaceRecognition.Recognizer import recognizer
from rest_framework.permissions import IsAdminUser, AllowAny

import os
import sys
import datetime as dt


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
    
    # Parse response as json and get list of timetables from it.
    timetable_id_list = data["timetable_period"]
    # print(data)
    # print(timetable_id_list)

    custom_header = {"Authorization": request.headers["Authorization"]}

    # Stores the list of timetable objects that were not found.
    # Return this list to user and ABORT??
    timetable_not_found = []

    set_of_sections = set()
    set_of_students = set()
    student_username_id = {}
    student_id_username = {}
    student_id_section = {}
    section_id_timetable_id = {}
    timetable_id_timetable_details = {}
    all_students = set()

    for timetable_id in timetable_id_list:
        print("\n\n############################################")
        print("timetable_id: {0}".format(timetable_id))

        # TODO: Make it async
        r = rq.get("http://127.0.0.1:8001/timetable_periods/{0}/".format(timetable_id),
                    headers=custom_header)

        if r.status_code == 404:
            timetable_not_found.append(timetable_id)
            continue
                
        elif r.status_code != 200:
            print("Error:" + r.text)

            # str.find() returns -1 if NOT FOUND.
            if r.headers['Content-Type'].find("json") != -1:
                print("Error: ")
                print(r.text)
                # print(r.encoding)
                # print(r.headers)
                print(json.dumps(json.loads(r.text), indent=4))

                return response.Response(status = r.status_code, data=json.loads(r.text))

            return response.Response(status=r.status_code, data={"error": "request to db-api returned an html page."})

        # Parse response as json and get section id from it.
        timetable_object = json.loads(r.text)
        print("\n\n## timetable_object: ")
        print(timetable_object)

        section_id = timetable_object["subject"]["section"]

        print("section id: {0}".format(section_id))
        set_of_sections.add(int(section_id))
        section_id_timetable_id.update({int(section_id): timetable_object["id"]})

        detail_str = timetable_object["subject"]["title"] + " " + timetable_object["section"] + " " + timetable_object["time"]
        timetable_id_timetable_details.update({int(timetable_object["id"]): detail_str})


    print("\n\n############################################")
    print("List of timetable objects sent by user that do not exist:")
    print(timetable_not_found)
    print("\n")

    print(set_of_sections)

    for section_id in set_of_sections:
        # Get list of all students in the section(s)
        custom_header = {"Authorization": request.headers["Authorization"]}
        url = "http://127.0.0.1:8001/students/?section={0}".format(section_id)
        # print(url)
        student_list_req = rq.get(url, headers=custom_header)

        if student_list_req.status_code == 200:
            student_list_json = json.loads(student_list_req.text)
            student_list = json.dumps(student_list_json, indent=4)
            
            # Get name of section.
            section_name = ""
            # print(len(student_list_json))
            if len(student_list_json):
                section_name = student_list_json[0]['section']['name']

            # print(student_list)

            print("Name of students in section `{0}`".format(section_name))
            for x in student_list_json:
                print(x["user"]['first_name'] + " " + x["user"]["last_name"])
                
                set_of_students.add(int(x["id"]))
                all_students.add(int(x["user"]["username"]))
                student_id_section.update({int(x["id"]): section_id})
                # Make hash table of student id and username.
                student_username_id.update({int(x["user"]["username"]) : int(x["id"])})
                student_id_username.update({int(x["id"]): int(x["user"]["username"])})

    # print(set_of_students)
    print("\n")

    # Turn on camera and identify students.
    identified_students = recognizer()
    # identified_students = {11620005, 1162222}

    # Present students are instersection of identified students and all_students
    present_students = identified_students.intersection(all_students)

    absent_students = all_students.difference(present_students)

    # print("student_username_id: ", end=" ")
    # print(student_username_id)

    # print("All Students: ", end=" ")
    # print(all_students)
    
    # print("Present Students: ", end=" ")
    # print(present_students)
    
    # print("Absent Students: ", end=" ")
    # print(absent_students)

    # student:section
    # section:timetable

    # print(student_id_section)
    # print(section_id_timetable_id)

    present_student_timetable = {}
    absent_student_timetable = {}

    print("Present Students")
    for p in present_students:
        student_id = student_username_id[p]

        section_id = student_id_section[student_id]

        for sec_id in section_id_timetable_id:
            if int(sec_id) == section_id:
                timetable_id = section_id_timetable_id[sec_id]

                present_student_timetable.update( {student_id: timetable_id})
                print(str(student_id) + ": " + str(timetable_id))


    print("Absent Students")
    for a in absent_students:
        student_id = student_username_id[a]
        section_id = student_id_section[student_id]

        # print("section:" + str(section_id))

        for sec_id in section_id_timetable_id:
            if int(sec_id) == section_id:
                timetable_id = section_id_timetable_id[sec_id]
        
                absent_student_timetable.update( {student_id: timetable_id})
                print(str(student_id) + ": " + str(timetable_id))

    print(present_student_timetable)
    print(absent_student_timetable)
    present_students = len(present_student_timetable)
    absent_students = len(absent_student_timetable)


    # Marking present_students present.
    for student_id in present_student_timetable:
        timetable_id = present_student_timetable[student_id]
        value = 1
        date = dt.datetime.now().date()

        data={
            "value": value,
            "date": str(date),
            "student": student_id,
            "timetable_period": timetable_id
        }

        custom_header = {"Authorization": request.headers["Authorization"]}

        r = rq.post("http://127.0.0.1:8001/attendance/", headers=custom_header, data=data)

        # print(r.text)

    # Marking present_students present.
    for student_id in absent_student_timetable:
        timetable_id = absent_student_timetable[student_id]
        value = 0
        date = dt.datetime.now().date()

        data={
            "value": value,
            "date": str(date),
            "student": student_id,
            "timetable_period": timetable_id
        }

        custom_header = {"Authorization": request.headers["Authorization"]}

        r = rq.post("http://127.0.0.1:8001/attendance/", headers=custom_header, data=data)

        # print(r.text)

    response_data = {   
        "success": "Attendance recorded successfully", 
        "students_present": present_students, 
        "students_absent": absent_students
        }

    return response.Response(status=200, data=response_data)