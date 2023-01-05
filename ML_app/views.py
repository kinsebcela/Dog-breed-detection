from django.shortcuts import render
from django.views import View
from django.http import HttpResponseRedirect
from aiprediction import *
# Create your views here.

def store_file(file):
    with open('ML_app/static/user_data/image.jpg', 'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)

def get_prediction():
    custom_image_path = ['ML_app/static/user_data/image.jpg']
    print('Image retrieved...')
    custom_data = create_data_batches(custom_image_path, test_data=True)
    print('Creating Data Batches...')
    custom_preds = loaded_full_model.predict(custom_data)
    print('Making predictions...')
    custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
    for dog_breed in custom_pred_labels:
        dog_breed = dog_breed.replace('_', ' ').title()
    return dog_breed


class UploadFile(View):
    def get(self, request):
        prediction = get_prediction()
        return render(request, 'ML_app/file-upload.html', {'prediction': prediction})

    def post(self, request):
        store_file(request.FILES['image'])
        return HttpResponseRedirect('/')