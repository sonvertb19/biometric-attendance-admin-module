from django.contrib import admin
from django.urls import path, include
from CreateStudentDataset import views as csd_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('createStudentDataset/', csd_views.create_student_dataset, name="create_student_dataset"),
    path('trainModel/', csd_views.train_model, name="train_model"),
    path("mark_attendance/", csd_views.mark_attendance, name="mark_attendance"),
]
