from flask import Flask, render_template, url_for, redirect
from flask_wtf import FlaskForm
import os
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
import numpy as np
from glob import glob
import cv2
from flask import request
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model




app = Flask(__name__)
app.config["SECRET_KEY"] = "AComplicat3dText."


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
Resnet50_model = load_model("trained_network")
Resnet50_model_detector = ResNet50(weights = "imagenet")

dog_names = []

with open("dog_categories.txt", encoding="utf-8") as f:
    for line in f.readlines():
        dog_names.append(line)
dog_names = [str(breed_name[:-1]) for breed_name in dog_names]
#dog_names = [item[20:-1] for item in sorted(glob("../dogImages/train/*/"))]
    





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
    #img = image.load_img(img_path, target_size=(224, 224))
    #plt.imshow(img)
    #plt.show()
    string = "For me this photo shows: {}".format(dog_names[np.argmax(predicted_vector)])
    return string


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if  len(faces) > 0:
        return True
    else:
        return False
    
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    if ((prediction <= 268) & (prediction >= 151)): 
        return True
    else:
        return False
    
    
def classifier(img_path,
               face_detector = face_detector,
               dog_detector = dog_detector,
               breed_detector = Resnet50_predict_breed):
    
    if dog_detector(img_path) == False and face_detector(img_path) == False:
            string = "This is neither a human nor a dog. Sorry I cannot define what it is."
           
           
    elif face_detector(img_path) == True and dog_detector(img_path) == False:
            breed_name = breed_detector(img_path).split("shows a")[1]
            string = "This person looks totally like {}".format(breed_name)
            
            current_breed = [substring for substring in os.listdir("static")
                       if breed_name[1:] in substring][0]
           
            img2 = image.load_img(os.path.join("static",current_breed))
            to_compare = secure_filename("to_compare.jpg")
            img2.save(os.path.join(app.root_path, "static", to_compare))
    
    elif dog_detector(img_path) == True:
            string = breed_detector(img_path)
    
    return string





class DogForm(FlaskForm):
    photograph = FileField("Click here to choose a file", validators = [FileRequired(), FileAllowed(["jpg", "png", "jpeg"], 
                                                                                    "Accepted are only the .jpg, .jpeg and .png files.")])





@app.route('/', methods=["POST", "GET"])
def index():
    
    dog = DogForm()

    
    if dog.validate_on_submit():
 
        f = dog.photograph.data
        to_predict = secure_filename("to_predict.jpg")
        f.save(os.path.join(app.root_path, "static", to_predict))     
        
        return redirect(url_for(("result")))

    return render_template("index.html", dog=dog)
    

@app.route("/result.html", methods=["POST", "GET"])
def result():
    string = str((classifier("static/to_predict.jpg")))
    return render_template("result.html", value = string)



if __name__=='__main__':
    app.run()



