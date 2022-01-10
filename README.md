# Udacity_project_2_DisasterResponse
Using Natural language processing to analyse and classify social media feeds to better respond to natural disasters
 

The project looks at way in which communication can be improved during a disaster situation. Whenever there’s a disaster incidents, the most common form of information sharing is via social media however it can be really challenging to really pickup information related to the actual disaster from the rest of the messages on social media.

To resolve this challenge, a model will be built to classify messages into 36 pre-defined categories which would make it easier to communicate with the relevant disaster relief agency by mapping the emergency to the appropriate services. These include aid related, medical help, fire, earthquakes etc.

The data used in the project has been provided by a company called ‘Figure Eight’. The main components of the project include:
•	ETL pipelines
•	Machine learning pipelines
•	Presenting findings on a web app

the components of the web app have been provided by Udacity, therfore the project will leverage on this existing app to further deploy additional visualizations 


FILE DESCRIPTION

        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- Preparation
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- ETL_Preparation.db
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
                |-- README
          |-- README
          
          
ISTALLATION
 
The following libraries must be installed to successfully run the files:

numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask 

INSTRUCTIONS

1. Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/
 
Licensing, Authors, Acknowledgements

Special thanks to Figure Eight for providing the data set.
many thanks to Udacity for designing and structuring the project for learning purposes.

