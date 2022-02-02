# DogBreeds_Classifier
Application for classifying dog breeds based on the convolutional neural networks

## Problem Statement and General Overview of the Project
The "DogBreeds_Classifier" is my capstone project which is the final submission in Udacity Data Scientist Nanodegree program.
It uses the concepts of convolutional neural networks (short CNN) for solving a dog breed classification problem.
The algorithms beeing implemented in this application are trained via supervised learning methods.
Below one can find the overview of the training, validation and test data set:

### The training data set
![image](https://user-images.githubusercontent.com/64994740/152216245-5303f3c9-dc56-4973-ab93-89dab1745647.png)
### The validation data set
![image](https://user-images.githubusercontent.com/64994740/152216419-4521f267-12d0-48df-94fa-0cea1134c25a.png)
### The test data set
![image](https://user-images.githubusercontent.com/64994740/152216401-1ccfbb11-64db-4a69-b609-33b7ea93941e.png)



# Analysis of the USA Road Accidents
### Descriptive and Inferential Statistics Conducted on the Kaggle's US Road Accidents Data Set.
***Project includes the following files:***  
***- US_RoadAccidents.ipynb -> jupyter notebook file containing whole analysis***  
***- US_Accidents_Dec20.csv -> csv file with records as of Decemeber 2020***

#### The motivation of the project is two-branch. In the first part I try to answer questions posed with regard to USA accidents' hot spots. The objective is to try to figure out, what kind of characteristics define an accident hot spot and which roads among all routes in USA seem to be most dangerous. In the second part I deal with the supervised learning model, were I try to predict the severity of an accident (in terms of traffic obstructions)

***This project serves in the first line the fulfillment of the exercise requirements of the first part of the Udacity's "Data Scientist" Nanodegree. Nevertheless, the data set used for this exercise contains actuall records and can be used to drag insights leading to real applications. 
The data set can be retrived from the following link (acutal version) https://www.kaggle.com/sobhanmoosavi/us-accidents  
For the analysis in the attached jupyter notebook file the version number 8 was used.<br/><br/><br/><br/>
Acknowledgements:  <br/><br/>
"Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.  
"Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM", 2019.***  <br/><br/><br/><br/>

### The code is written in Python 3.7.3. Required libraries to run the project:

Python basic libraries:
- math
- random  <br/><br/>

Python data science libraries:
- pandas
- numpy

Python visualistion libraries:
- matplotlib
- seaborn <br/><br/>

Python machine learning libraries:
- sklearn <br/><br/>

Python geospatial data science libraries:
- geopandas
- geoplot (Python 3.6+ only!) <br/><br/>  

Installation is possible via Anaconda Prompt. For all libraries the command `conda install "library_name"` will work. The geoplot has to be installed over the Forge Channel:  `conda install geoplot -c conda-forge` <br/><br/><br/>

## Insights Summary: </br></br>
### Accidents' Hot Spots
- I was able to distinguish three roads with accident pile ups. Of course these are not the only ones, however they seem to be outstanding from the others.
Please refer to the following picutres:  
![insight1](https://user-images.githubusercontent.com/64994740/127754365-1f627c56-3c20-411f-9cac-7aa0751a3902.png)



</br>
Amenities and environmental conditions connected to the accidents' frequency:<br/>

![Insight2](https://user-images.githubusercontent.com/64994740/127754371-29e29c15-12a0-4f04-a2a4-b4b812acd35c.png)
<br/>

![Insight3](https://user-images.githubusercontent.com/64994740/127754376-eb841206-bebf-4b50-ae83-7d541fe7a975.png)


### Application of Neural Network for predicting the Severity of an Accident
- Producing the algorithm to predicit the accidents' severity is hampered due to strongly unbalanced number of different labels in the data set. Since the mode (severity class 2) makes around 70% percent of all records, an accuracy level of around 75% (obtained by model fitted onto original data) does not bring much. One could get nearly the same results just by picking up the mode. Less represented classes could not be predicted properly. Therefore, the data set has been balanced so that all of the classes contained the same number of records. Such an approach enhenced the models capability to predict among less numerous classes, however got worse while predicting class 2 beeing now underrepresented. Please compare the results below:</br></br></br></br></br>
![balanced_unbalanced_comparison](https://user-images.githubusercontent.com/64994740/127754238-9c2915cb-38b1-4c0e-9dbf-44879506e173.png)

Realted article on Medium:
https://medium.com/@niedzielskijacek/usa-road-accidents-hot-spots-and-severity-5a0ef2541521
