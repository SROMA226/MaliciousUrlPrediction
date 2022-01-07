from pathlib import Path

class Config:
    RANDON_SEED = 28 # Seed
    TEST_SIZE = 0.2
    ASSETS_PATH = Path("./assets")
    ORIGINAL_DATASET_FILE_PATH = ASSETS_PATH / "original_dataset" / "file-urls.csv" # Dataset original
    DATASET_PATH = ASSETS_PATH / "data" # Dossier pour notre dataset
    FEATURES_PATH = ASSETS_PATH / "features" # Dossier pour les features 
    MODELS_PATH = ASSETS_PATH / "models" # Dossier pour les modeles
    METRICS_FILE_PATH = ASSETS_PATH / "metrics.json" # Fichiers pour les mesures