from flask import Flask, render_template, url_for, redirect, flash 
from flask_wtf import FlaskForm
from grpc import ssl_channel_credentials
from wtforms import StringField, IntegerField, BooleanField, DateField
from wtforms.validators import DataRequired, Length, ValidationError, Email
import os
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
import numpy as np
from glob import glob
import cv2
from flask import request


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model




app = Flask(__name__)
app.config["SECRET_KEY"] = "AComplicat3dText."


face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')


bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
Resnet50_model = load_model("trained_network")
Resnet50_model_detector = ResNet50(weights = "imagenet")


dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]
    





def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(Resnet50_model_detector.predict(img))



def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False, pooling="avg" ).predict(preprocess_input(tensor))

def Resnet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if  len(faces) > 0:
        return True
    
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    if ((prediction <= 268) & (prediction >= 151)): 
        return True
    else:
        return False
    
    
    



def classifier(img_path):
    if (dog_detector(img_path) == False and face_detector(img_path) == False):
        return "This is neither a human nor a dog"
    
    elif (face_detector(img_path) == True and dog_detector(img_path) == False):
        return "This person looks totally like {}".format(Resnet50_predict_breed(img_path))

    elif dog_detector(img_path) == True:
        return Resnet50_predict_breed(img_path)
    
    
    






string = str((classifier("../images/Bucky_4.jpg")))


class DogForm(FlaskForm):
    
    photograph = FileField("Click me to choose a file", validators = [FileRequired(), FileAllowed(["jpg", "png"], "Accepted are only the .jpg and .png files.")])





@app.route('/', methods=["POST", "GET"])
def index():
    
    dog = DogForm()
    
    if request.method == "POST" and dog.validate_on_submit():
        print("Proper content!")
        f = dog.photograph.data
        filename = f.filename
        f.save(os.path.join(app.root_path, "static", filename))       
        return redirect(url_for("result"))

    elif request.method == "POST" and dog.validate_on_submit() == False:
        return redirect(request.referrer)

    print("rendering")
    
    return render_template("index.html", dog = dog)
    
  
    
    
    
    

"""

    if request.method == "GET" and dog.validate_on_submit() == False:
        return render_template("index.html", dog = dog)
    
    if request.method == "POST" and dog.validate_on_submit():
        return redirect(url_for("result"))
    
    if request.method == "POST" and dog.validate_on_submit() == False:
            
        return render_template("index.html", dog = dog)
"""


@app.route("/result.html", methods=["POST", "GET"])
def result(string = string):
   
    return render_template("result.html", value = string)



if __name__=='__main__':
    app.run()










