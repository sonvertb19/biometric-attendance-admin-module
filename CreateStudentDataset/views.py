from rest_framework import response
from rest_framework.decorators import api_view
import requests as rq
import json
from FaceRecognition.Create_dataset import run_script
from FaceRecognition.Train import train_faces


@api_view(["GET"])
def create_student_dataset(request):

    # Making a request to the BasDB_API server.
    #   Adding Custom Auth Header.
    custom_header = {'Authorization': request.headers['Authorization']}

    roll_number = request.data['user']['username']
    print(roll_number)

    # TODO: (FOR YASH ARORA)If run_script is not completed properly, do not send request to the server.
    """
        Return a value from run_script and act correspondingly. 
        Something like: If all 180 images have not been created and written to the disk,
        return an error dictionary.
        Else return True. 
    """
    run_script(roll_number=roll_number)

    r = rq.post("http://127.0.0.1:8001/students/", json=request.data, headers=custom_header)

    print("Request with the above data made to the server.")
    print("Response: ", end=" ")
    print(r.text)
    print("status_code: ", end=" ")
    print(r.status_code)

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
