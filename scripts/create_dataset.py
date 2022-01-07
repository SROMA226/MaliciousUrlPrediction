# Telecharger le dataset depuis un GDrive
# Split en train et test
# Enregistrer dans "assets/data"

from numpy.core.defchararray import index
from scipy.sparse.construct import rand
import gdown
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import Config

# Set seed
np.random.seed(Config.RANDON_SEED)

# Creer les dossier dont on a besoin dans ce script
# ./assets/original_datasets & ./assets/data
Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Telecharge notre fichier
gdown.download(
    "https://drive.google.com/uc?id=1AuOCki1-TLLAGElNQEBL7n9Rg8Rd5qtX",
    str(Config.ORIGINAL_DATASET_FILE_PATH)
)

# dataframe
df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH))

df_train, df_test = train_test_split(
    df, test_size=Config.TEST_SIZE, 
    random_state=Config.RANDON_SEED
)

df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)