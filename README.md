# DogBreeds_Classifier
Application for classifying dog breeds based on the convolutional neural networks

## Project Overview 
The "DogBreeds_Classifier" is my capstone project which is the final submission in the Udacity Data Scientist Nanodegree program.
It uses the concepts of convolutional neural networks (short CNN) for solving a dog breed classification problem (multiclass classification - 133 output classes)
The algorithms beeing implemented in this application are trained via supervised learning methods.
Below one can find the overview of the training, validation and test data sets which were used to train the CNN network from scratch.
All the data has been provided by Udacity. At some stages of my project I have used the stanford dog data set 
`https://www.kaggle.com/jessicali9530/stanford-dogs-dataset`, but at the end of the day I made use only of the data presented below:

### The training data set
![image](https://user-images.githubusercontent.com/64994740/152216245-5303f3c9-dc56-4973-ab93-89dab1745647.png)
### The validation data set
![image](https://user-images.githubusercontent.com/64994740/152216419-4521f267-12d0-48df-94fa-0cea1134c25a.png)
### The test data set
![image](https://user-images.githubusercontent.com/64994740/152216401-1ccfbb11-64db-4a69-b609-33b7ea93941e.png)

The architecture summary of the CNN is visible in the next picture:

![Bez tytułu](https://user-images.githubusercontent.com/64994740/152219184-c59c6727-8a1e-4992-8df3-59e879063b7b.png)

However the final model is based on the transfer learning. The bottleneck features used for the final CNN have been provided by Udacity. In case of this very project
I have decided to pick the Resnet50 model which is available as one of the keras applications and add some additional layers. For more details please examine the jupyter notebook attached in this repository.


## Problem Statement and Metrics
As mentioned above I have tried to build the CNN architecture from scratch firstly. I have experimented with several architectures beeing of different depths as well as with ensemble models. The test performance achieved with some data augmentation and reasonable computation time was around 15%. The CNN structure was definetely too shallow, which could be especially confirmed by 18% train accuracy. Please see `dog_app.ipynb` for details.
Since the train performance (accuracy) of the "made from scratch network" was not satisfactory and one would need to make much deeper architecture (which means longer fitting time) to prevent uderfitting, I have decided to make use of transfer learning. To accomplish that, I have used the the keras Resnet50 application which is based on imagenet dataset. The achieved accuracy for this model by this particular commit is around 80%. I have chosen the accuracy as a metrics despite the fact that the data is quite inbalanced. Since the problem domain is a concept of fun application, it is not so important to get high recall or precision as in case of more serious topics (like for example account fraud classification). For more details regarding analysis steps and conclusions please refer to the jupyter notebook beeing part of this repository `dog_app.ipynb`. Please install the ipywidgets to see the visualisations!

## Data Exploration and Data Visualisation
The data exploration and visualisation steps are presented in detail in `dog_app.ipynb` (ipywidgets library required!). However I would like to emphasize it here once more, that the data used for building the CNN models from scratch is imbalanced. Bigger problem is however low amount of data. Below the summary:<br>
![image](https://user-images.githubusercontent.com/64994740/152435343-83554706-6901-4189-ba1d-638c58a2f86c.png)

It means only around 62 photos per breed. Taking into consideration the fact, that some breeds are very simillar to each other, low accuracy shouldn't be a surprise. On the other hand the Resnet50 model is trained on millions of different images from imagenet. This is one for the reasons why it performs much better.

## Data Preprocessing
In case of this project no preprocessing steps were necessary. All of the data provided by Udacity was "ready to learn". The only data preprocessing steps were those connected to data augmentation, but it shall be rather defined as model tuning, than pure wrangling.

##Implementation
After successful training of the CNN based on transfer learning one has to build a classifier. Basically we can divide the classifier into three parts. More details are available in the `dog_app.ipynb`, bellow the summary:<br>
![image](https://user-images.githubusercontent.com/64994740/152436393-ff20b657-b924-4561-b96b-5d11645ab92b.png)







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
