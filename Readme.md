# Project Introduction
In this course, you've learned and built on your data engineering skills to expand your opportunities and potential as a data scientist. In this project, you'll apply these skills to analyze disaster data from Appen(opens in a new tab) (formerly Figure 8) to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!


# Data Source
Disaster data from Appen(opens in a new tab) (formerly Figure 8) 


# License: MIT License


# Pre-condition
Download sample datasets and relevant codes from Udacity.


## Step 1 : ETL Pipeline
Complete ETL Pipeline with loading message and category data, data process and save out data into SQLite database.



## Step 2 : ML Pipeline
Loaded data from SQLite database,train AI model with Pipeline and improve the performance with GridSearchCV.


## Step 3 :
Refer to "ETL Pipeline Preparation.ipynb" file to update "process_data.py" with packaging relevant codes to ensure data loading,processing and saving out to SQLite..
Refer to "ML Pipeline Preparation.ipynb" file to update "train_classifier.py" with packaging relevant codes to ensure data loading, training, evaluating performance and save out AI model.

## Step 4 : 
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

