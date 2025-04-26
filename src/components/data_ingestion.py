# Importing the os module to interact with the operating system (file paths, directories)
import os

# Importing the sys module to interact with the Python runtime environment (used in custom exception handling)
import sys

# Importing a custom exception class defined in the project
from src.exception import CustomException

# Importing a custom logging module for logging information during code execution
from src.logger import logging

# Importing pandas library for data manipulation and analysis
import pandas as pd

# Importing the function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Importing dataclass decorator to create classes mainly for storing data without writing boilerplate code
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

from src.components.data_transformation import DataTransformationConfig

# Defining a configuration class to store the file paths for train, test, and raw data
@dataclass
class DataIngestionConfig:
    # Defining the path where training data will be stored
    train_data_path: str = os.path.join('artifacts', "train.csv")
    # Defining the path where testing data will be stored
    test_data_path: str = os.path.join('artifacts', "test.csv")
    # Defining the path where raw input data will be stored
    raw_data_path: str = os.path.join('artifacts', "data.csv")


# Creating a class for Data Ingestion process
class DataIngestion:
    def __init__(self):
        # Instantiating the DataIngestionConfig class to access file paths
        self.ingestion_config = DataIngestionConfig()

    # Defining a method to start the data ingestion process
    def initiate_data_ingestion(self):
        # Logging an info message to indicate the start of data ingestion
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the dataset CSV file into a pandas dataframe
            df = pd.read_csv(r'src\notebook\data\stud.csv')
            # Logging that dataset has been successfully read
            logging.info('Read the dataset as dataframe')

            # Creating the directories if they do not exist to store the files (train/test/raw)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving the original/raw data to the specified raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Logging the start of train-test splitting
            logging.info("Train test split initiated")
            # Splitting the data into training (80%) and testing (20%) sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the training set into the specified path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Saving the testing set into the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Logging that data ingestion has been completed successfully
            logging.info("Ingestion of the data is completed")

            # Returning the paths of the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raising a custom exception in case of any error during the process
            raise CustomException(e, sys)
        

# The below code will only execute if this file is run directly (not imported)
if __name__ == "__main__":
    # Creating an object of the DataIngestion class
    obj = DataIngestion()
    # Initiating data ingestion and receiving train and test data paths
    train_data, test_data = obj.initiate_data_ingestion()

    # The below commented code is meant for further steps like Data Transformation and Model Training
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr, test_arr))