import pickle # Serialiser des objets (y comporis des modeles)
import json 

import pandas as pd
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, classification_report

from config import Config

X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))
y_test = y_test.to_numpy().ravel()

# Restaurer le modèle
model = pickle.load(open(str(Config.MODELS_PATH / "model.pk"), mode='rb'))

y_pred = model.predict(X_test)
test_fscore = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
test_precision = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
test_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
test_recall = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')

#target_names = ['benign', 'defacement', 'malware','phishing']
#print(classification_report(y_test, y_pred, target_names=target_names))

with open(str(Config.METRICS_FILE_PATH), mode='w') as f:
    json.dump(dict(f1_score=test_fscore, precision=test_precision, recall_score=test_recall,accuracy_score=test_accuracy), f)
