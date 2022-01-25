from tkinter import font
from flask import Flask, render_template, url_for
from flask_wtf import FlaskForm
from grpc import ssl_channel_credentials
from wtforms import StringField, IntegerField, BooleanField, DateField
from wtforms.validators import DataRequired, Length, ValidationError, Email
import os
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from datetime import date
from wtforms.widgets import FileInput, PasswordInput, SubmitInput

app = Flask(__name__)
app.config["SECRET_KEY"] = "AComplicat3dText."

class Book:
    def __init__(self, title, amount, available, email, offer_date):
        self.title = title
        self.amount = amount
        self.available = available
        self.email = email
        self.offer_date = offer_date


class DogForm(FlaskForm):
    
    photograph = FileField("Click me to choose a file", validators = [FileRequired(), FileAllowed(["jpg", "png"], "Accepted are only the .jpg and .png files.")])


@app.route('/', methods=["POST", "GET"])
def index():

    dog = DogForm()
    
    if dog.validate_on_submit():

        f = dog.photograph.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.root_path, "static", filename))
        return render_template("result.html")
        
    return render_template("index.html", dog = dog)




if __name__=='__main__':
    app.run()










