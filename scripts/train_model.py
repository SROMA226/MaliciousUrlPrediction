import pickle # Serialiser des objets (y comporis des modeles)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier 

from config import Config

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

# Entrainement du model
model = DecisionTreeClassifier(max_depth=18)
model.fit(X_train, y_train.to_numpy().ravel()) # e.g. array([[1], [0]]).ravel() = array([1, 0])

# Enregisrement du model
pickle.dump(model, open(str(Config.MODELS_PATH / "model.pk"), mode='wb'))
