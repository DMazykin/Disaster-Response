# Disaster Response Pipeline Project

## Summary:

The project is a set of tools for analyzing messages, received during Natural disasters events, like earthquakes, storm or floods.

Using Machine learning technics and already collected data, this project helps to identify a message and assign appropriate categories. Such categories allow to link a received message with an proper organization, which can help people in difficult situation.

There are three main parts in this project:
- ETL pipeline is for cleaning and preparing data for the next step.
- ML pipeline use the data from the previous step and creates a Machine learning model with the best parameters for message categorization. 
- Web-app: a front-end of this project aimed to visualize data (used for ML model) and predicting categories for a new message.

As initial dataset, I used preheated and classified data from Figure Eight: https://www.figure-eight.com/dataset/combined-disaster-response-data/

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## Main files in the repository

- `data/process_data.py` - ETL pipeline used to prepare data for ML model building and save the date SQL file.
- `models/train_classifier.py` - ML pipeline used to train a model with the best parameters and save the trained model to pickle file
- `app/run.py` - Script to start a web-app for data visualization and message classification. This script uses the following templates:
  - `app/templates/master.html` - main template
  - `app/templates/go.html` - template for classification output 