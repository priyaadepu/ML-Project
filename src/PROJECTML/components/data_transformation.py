import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.PROJECTML.utils import save_object
from src.PROJECTML.exception import CustomException
from src.PROJECTML.logger import logging
import os

# Constants for Column Names
NUMERICAL_COLUMNS = ["writing score", "reading score"]  # fixed spacing in column names
CATEGORICAL_COLUMNS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]
TARGET_COLUMN = "math score"

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation (scaling and encoding).
        '''
        try:
            # Create numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            # Create categorical pipeline
            cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
            ])

            # Log columns being used
            logging.info(f"Categorical Columns: {CATEGORICAL_COLUMNS}")
            logging.info(f"Numerical Columns: {NUMERICAL_COLUMNS}")

            # Preprocessing pipeline with both numerical and categorical transformations
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, NUMERICAL_COLUMNS),
                    ("cat_pipeline", cat_pipeline, CATEGORICAL_COLUMNS)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files completed")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            # Split the train and test datasets into input and target features
            input_features_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Applying preprocessing on training and test dataframes")


            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Combine transformed features with target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"saved preprocessing object")

            # Ensure artifacts directory exists
            # os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            # Save the preprocessing object
            # logging.info(f"Saving preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return final arrays and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
