import pandas as pd
import os

dataFolder = "./data"

ot_odr_filename = os.path.join(dataFolder, "OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

equipements_filename = os.path.join(dataFolder, 'EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename,
                             sep=";")

print(equipements_df.head())
print(ot_odr_df.columns)

# On commence par faire un merge entre les deux tables
merged_df = pd.merge(ot_odr_df, equipements_df, on="EQU_ID")
print(merged_df.columns)


# On separe les donnees en deux jeux de donnees : Un pour l'entrainement, l'autre pour le test
from sklearn.model_selection import train_test_split
import pyAgrum as gum

train_data, test_data = train_test_split(merged_df, test_size=0.2, random_state=42)

print("Training set size:", len(train_data))
print("Testing set size:", len(test_data))

# On cree un BN avec pyAgrum
bn = gum.BayesNet("modèle simple")




