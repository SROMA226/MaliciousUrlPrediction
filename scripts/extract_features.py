# Considerer train.csv et test.csv
# Extraire les caracteriques qui nous intereste
# Enregistrer

import pandas as pd

from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
df_test = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))

features = ['length_of_url', 'number_of_letters', 'number_of_digits',
       'count_of_dotcom', 'count_of_codot', 'count_of_dotnet',
       'count_of_forward_slash', 'count_of_upper_case', 'count_of_lower_case',
       'count_of_dot', 'count_of_dot_info', 'count_of_https',
       'count_of_www_dot', 'count_of_not_alphanumeric', 'count_of_percentage',
       'percentage_to_length_ratio','forwards_slash_to_length_ratio']

type_map = {
    'benign': 0, 
    'defacement': 1,
    'malware': 2, 
    'phishing': 3
}

# supprimer toutes les lignes dupliqueées
df_train.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)


def extract_features(df):

    length_of_url = [] # longueur (nombre de caractères) de l'URL
    number_of_letters = [] # nombre de caractères alphabetiques
    number_of_digits = [] # nombre de caractères numériques
    count_of_dotcom = [] # compter le nombre de '.com' dans une URL
    count_of_codot = [] # compter le nombre de '.co.' dans une URL
    count_of_dotnet = [] # compter le nombre de '.net' dans une URL
    count_of_forward_slash = [] # compter le nombre de '/' dans une URL
    count_of_percentage = [] # compter le nombre de '%' dans une URL
    count_of_upper_case = [] # compter le nombre de caractères majuscules dans une URL
    count_of_lower_case = [] # compter le nombre de caractères minuscules dans une URL
    count_of_dot = [] # compter le nombre de '.' dans une URL
    count_of_dot_info = [] # compter le nombre de '.info' dans une URL
    count_of_https = [] # compter le nombre de 'https' dans une URL
    count_of_www_dot = [] # compter le nombre de 'www.' dans une URL
    count_of_not_alphanumeric = [] # compter le nombre de caractères non alpha-numeriques


    for item in df['url']:
        try:
            length_of_url.append(len(item))
        except:
            length_of_url.append(0)
            
        try:
            number_of_letters.append(sum(c.isalpha() for c in item))
        except:
            number_of_letters.append(0)
            
        try:
            number_of_digits.append(sum(c.isdigit() for c in item))
        except:
            number_of_digits.append(0)
        
        try:
            count_of_dotcom.append(item.count(".com"))
        except:
            count_of_dotcom.append(0)
            
        try:
            count_of_codot.append(item.count(".co."))
        except:
            count_of_codot.append(0)
            
        try:
            count_of_dotnet.append(item.count(".net"))
        except:
            count_of_dotnet.append(0)
            
        try:
            count_of_forward_slash.append(item.count("/"))
        except:
            count_of_forward_slash.append(0)
            
        try:
            count_of_percentage.append(item.count("%"))
        except:
            count_of_percentage.append(0)

        try:
            count_of_dot.append(item.count("."))
        except:
            count_of_dot.append(0)    
            
        try:
            count_of_upper_case.append(sum(c.isupper() for c in item))
        except:
            count_of_upper_case.append(0)
        
        try:
            count_of_lower_case.append(sum(c.islower() for c in item))
        except:
            count_of_lower_case.append(0)
            
        try:
            count_of_dot_info.append(item.count(".info"))
        except:
            count_of_dot_info.append(0)
            
        try:
            count_of_https.append(item.count("https"))
        except:
            count_of_https.append(0)
            
        try:
            count_of_www_dot.append(item.count("www."))
        except:
            count_of_www_dot.append(0)
            
        try:
            count_of_not_alphanumeric.append(sum(not c.isalnum() for c in item))
        except:
            count_of_not_alphanumeric.append(0)

    df['length_of_url'] = length_of_url
    df['number_of_letters'] = number_of_letters
    df['number_of_digits'] = number_of_digits
    df['count_of_dotcom'] = count_of_dotcom
    df['count_of_codot'] = count_of_codot
    df['count_of_dotnet'] = count_of_dotnet
    df['count_of_forward_slash'] = count_of_forward_slash
    df['count_of_upper_case'] = count_of_upper_case
    df['count_of_lower_case'] = count_of_lower_case
    df['count_of_dot'] = count_of_dot
    df['count_of_dot_info'] = count_of_dot_info
    df['count_of_https'] = count_of_https
    df['count_of_www_dot'] = count_of_www_dot
    df['count_of_not_alphanumeric'] = count_of_not_alphanumeric
    df['count_of_percentage'] = count_of_percentage

    ## Le ratio des caractères non alpha-numeriques
    df['not_alphanumeric_to_letters_ratio'] = df['count_of_not_alphanumeric']/df['number_of_letters']

    ## Le ratio du caractère '%'
    df['percentage_to_length_ratio'] = df['count_of_percentage']/df['length_of_url']

    ## Le ratio du caractère '/'
    df['forwards_slash_to_length_ratio'] = df['count_of_forward_slash']/df['length_of_url']

    ## Le ratio entre caractères majuscules et minuscules
    df['upper_case_to_lower_case_ratio'] = df['count_of_upper_case']/df['count_of_lower_case']
  
    # Transformer les valeurs de type d'urls en leur equivalent dans `type_map`
    def process_type(row):
        return type_map[row['type']]
        
    df['type'] = df.apply(lambda row: process_type(row), axis=1)

    return df[features]

train_features = extract_features(df_train)
test_features = extract_features(df_test)

# Enregistrement des features pour train et test
train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

# Enregistrement des labels (la colonne type de notre dataset) pour train et test
df_train.type.to_csv(str(Config.FEATURES_PATH / "train_labels.csv"), index=None)
df_test.type.to_csv(str(Config.FEATURES_PATH / "test_labels.csv"), index=None)
