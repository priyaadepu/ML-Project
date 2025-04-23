import os
import sys
import pickle
import pandas as pd
import numpy as np
import pymysql 
from dotenv import load_dotenv
from src.PROJECTML.exception import CustomException
from src.PROJECTML.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


load_dotenv()

# Load environment variables
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv('db')

# Read data from MySQL
def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established: {mydb}")
        df = pd.read_sql_query('SELECT * FROM students', mydb)
        print(df.head())
        return df

    except Exception as ex:
        raise CustomException(ex, sys)

# Save a Python object to file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

# Uncomment the following functions if you plan to use model evaluation

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
