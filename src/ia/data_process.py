import pandas as pd
import os


class DataProcess:
    def __init__(self):
        self.data_folder = "./data"
        self.ot_odr_df = None
        self.equipements_df = None
        self.data_df = None
        self.train_data = None
        self.test_data = None

    def load_data(self):
        ot_odr_filename = os.path.join(self.data_folder, "OT_ODR.csv.bz2")
        self.ot_odr_df = pd.read_csv(ot_odr_filename, compression="bz2", sep=";")

        equipements_filename = os.path.join(self.data_folder, "EQUIPEMENTS.csv")
        self.equipements_df = pd.read_csv(equipements_filename, sep=";")

    def merge_data(self):
        self.data_df = pd.merge(self.ot_odr_df, self.equipements_df, on="EQU_ID")

    def print_data_info(self):
        print("Data set size:", len(self.data_df))

    def run(self):
        self.load_data()
        self.merge_data()
        self.print_data_info()
