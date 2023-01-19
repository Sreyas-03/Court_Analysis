# Court_Analysis
Analysis of data filed in the Indian courts between the years 2010-2018
___
___

This is my brief analysis of data about the Indian courts and cases.

The dataset analysed has data ***about 80 million cases*** filed in the period 2010-18.

The datas include Gender of judges, petitioner, defendent, advocates; acts, sections under which case is filed;
Date of first, most recent, next and final hearings; disposition given on a case; the judge who handled the case, their career duration, etc.

Link to the database:
[1]:https://www.dropbox.com/sh/hkcde3z2l1h9mq1/AAB2U1dYf6pR7qij1tQ5y11Fa/csv?dl=0&subfolder_nav_tracking=1

___

## source code used for analysis

- analysis.py - contains the code used for analysing the data

- model_training.py - contains the program to train and test the models

- Insights.pdf - contains some of th einsights i have drawn from this data

___

## Models Trained

- disp_punished.h5 - It is a classifier that predicts whether a case is punishable or not. The ***accuracy of the model is ~88%***

- criminal_acts_sections.h5 - It is a classifier to predict if a case is criminal or not. The ***accuracy of the model is ~90%***

 ***(For more info about the models, look at Insights.pdf)***
 

