import logging
import pandas as pd
from project.code import main_program


class DataScienceProject:
    def __init__(self, path):
        self.path = path
        self.setup_logging()
        self.setup_pandas()

    def setup_logging(self):
        # Create a custom logger
        self.logger = logging.getLogger(__name__)

        # Set the logging level
        self.logger.setLevel(logging.ERROR)

        # Create handlers
        file_handler = logging.FileHandler('program_errors.log')
        console_handler = logging.StreamHandler()

        # Set logging level for handlers
        file_handler.setLevel(logging.ERROR)
        console_handler.setLevel(logging.ERROR)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def setup_pandas(self):
        pd.set_option('display.max_rows', None)

    def run(self):
        try:
            main_program(self.path, self.logger)
        except Exception as e:
            self.logger.error(f"Error running main_program: {e}")
            print(f"Error running main_program: {e}")


# Usage
if __name__ == "__main__":
    project = DataScienceProject('jobs_in_data.csv')
    project.run()
