# DogBreeds_Classifier
Application for classifying dog breeds based on the convolutional neural networks

## Problem Statement and General Overview of the Project
The "DogBreeds_Classifier" is my capstone project which is the final submission in Udacity Data Scientist Nanodegree program.
It uses the concepts of convolutional neural networks (short CNN) for solving a dog breed classification problem.
The algorithms beeing implemented in this application are trained via supervised learning methods.
Below one can find the overview of the training, validation and test data set which were used to train the CNN network from scratch.

### The training data set
![image](https://user-images.githubusercontent.com/64994740/152216245-5303f3c9-dc56-4973-ab93-89dab1745647.png)
### The validation data set
![image](https://user-images.githubusercontent.com/64994740/152216419-4521f267-12d0-48df-94fa-0cea1134c25a.png)
### The test data set
![image](https://user-images.githubusercontent.com/64994740/152216401-1ccfbb11-64db-4a69-b609-33b7ea93941e.png)

The architecture summary of the CNN is visible in the next picture:

![Bez tytułu](https://user-images.githubusercontent.com/64994740/152219184-c59c6727-8a1e-4992-8df3-59e879063b7b.png)



Since the train performance (accuracy) of this network was not satisfactory and one would need to make much deeper architecture (which means longer fitting time) to prevent uderfitting, I have decided to make use of transfer learning. To accomplish that, I have used the the keras Resnet50 application which is based on imagenet dataset. The achieved accuracy for this model by this particular commit is around 80%. I have chosen the accuracy as a metrics despite the fact that the data is quite inbalanced. Since the problem domain is a concept of fun application is is not so important to get high recall or precision as in case of more serious topics (for example account fraud classification). For more details regarding analysis steps and conclusions please refer to the jupyter notebook beeing part of this repository `dog_app.ipynb`. Please install the ipywidgets to see the visualisations!

## Application
The application is build on top of `Flask` framework. It makes use of the classification algorithm defined during abovementioned project, and predicts the dog breed from the uploaded photo. If a picture of an human is provided, the application will return the most resembling dog breed. If the photograph contains neither human nor dog (according to the classifier) an appropiate message will be returned.

**In order to run the application locally:**

1. Clone this repository
2. Make a new conda environment with Python 3.7
3. Navigate to the app folder
4. Type in Anaconda Prompt: `pip install -r requirements.txt`
5. When all the required modules are installed you can type the command `flask run`. The app will open on a local host. **It may be required to deactivate the debugger!**

## Jupyter notebook
The juypter notebook file `dog_app.ipynb` includes more details about analysis and conclusions. You will need to additionally install `pandas` and `seaborn` in order to run it
(ipywidgets are already in the requirements.txt file)

## Acknowledgments
Udacity Capstone Dog Project -> https://www.udacity.com/course/data-scientist-nanodegree--nd025
