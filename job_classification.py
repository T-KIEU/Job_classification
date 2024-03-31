# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:14:13 2023

@author: kieu_
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
import re
from imblearn.over_sampling import RandomOverSampler, SMOTEN # SMOTE pour les données numerical, et SMOTEN pour les données nominal



data = pd.read_excel(r"C:\Users\kieu_\OneDrive\Desktop\Project\Data Science\VietNguyen\ML_9-1\Datasets\final_project.ods", engine="odf", dtype=str)

# data.head()
#                                                title  ...                          career_level
# 0              Technical Professional Lead - Process  ...  senior_specialist_or_project_manager
# 1                    Cnslt - Systems Eng- Midrange 1  ...  senior_specialist_or_project_manager
# 2      SharePoint Developers and Solution Architects  ...  senior_specialist_or_project_manager
# 3  Business Information Services - Strategic Acco...  ...  senior_specialist_or_project_manager
# 4       Strategic Development Director (procurement)  ...                        bereichsleiter



# data.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 8074 entries, 0 to 8073
# Data columns (total 6 columns):
#  #   Column        Non-Null Count  Dtype 
# ---  ------        --------------  ----- 
#  0   title         8074 non-null   string
#  1   location      8074 non-null   string
#  2   description   8074 non-null   string
#  3   function      8074 non-null   string
#  4   industry      8074 non-null   string
#  5   career_level  8074 non-null   string
# dtypes: string(6)
# memory usage: 378.6 KB



# "location" column
# X_train["location"].head()
# 4107       Savannah, GA
# 938        New York, NY
# 8064    Saint Louis, MI
# 7096       Cranston, RI
# 5175      San Ramon, CA

# print(len(X_train["location"].unique()))
# 969 -> 969 valeurs uniques

# Comme c'est une variable nominale, on devrait utiliser OneHot mais 969 est un grand nb de valeurs, le vecteur sera trop long.
# C'est pourquoi on va splitter et prendre seulement les 2 caractères de l'état avant de faire OneHot

# Fonction de récupération des 2 caractères de l'état (à appliquer dans le split du train et test)
def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location) # regex pour prendre à partir de la virgule
    if len(result) > 0:
        return result[0][2:] # récupérer d'abord le résultat avec [0], puis prendre uniquement à partir du 2ème caractères
    else:
        return location

data["location"] = data["location"].apply(filter_location)
# data["location"].head()
# 0                TX
# 1                WA
# 2                TX
# 3    North Carolina
# 4                TX



target = "career_level"

X = data.drop(target, axis=1)
y = data[target]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print(y_train.value_counts())
# senior_specialist_or_project_manager      3470
# manager_team_leader                       2138
# bereichsleiter                             768
# director_business_unit_leader               56
# specialist                                  24
# managing_director_small_medium_company       3
# Name: career_level, dtype: int64

# print(y_test.value_counts())
# senior_specialist_or_project_manager      868
# manager_team_leader                       534
# bereichsleiter                            192
# director_business_unit_leader              14
# specialist                                  6
# managing_director_small_medium_company      1
# Name: career_level, dtype: int64




### Over sampling (générer de nouvelle données basées sur les données existantes -> objectif : dataset soit moins imbalanced)
    # ATTENTION, cette étape est à faire après avoir splitté en train et test, pour avoir les mêmes données à la fois dans train et dans test (data leakage)
# ros = RandomOverSampler(random_state=42)
# print(y_train.value_counts())
# senior_specialist_or_project_manager      3470
# manager_team_leader                       2138
# bereichsleiter                             768
# director_business_unit_leader               56
# specialist                                  24
# managing_director_small_medium_company       3
# Name: career_level, dtype: int64

# X_train, y_train = ros.fit_resample(X_train, y_train) # uniquement pour le train
# print(y_train.value_counts())
# manager_team_leader                       3470
# senior_specialist_or_project_manager      3470
# bereichsleiter                            3470
# specialist                                3470
# director_business_unit_leader             3470
# managing_director_small_medium_company    3470
# Name: career_level, dtype: int64


# On n'a pas besoin de faire monter les plus petites classes au même niveau que la plus grande.
    # L'idée n'est pas de faire que dataset soit complètement balanced, mais l'idée est de le rendre moins imbalanced.
    # Cela permettra que le modèle apprenne et prédise mieux les petites classes
    # Avec sampling_strategy, on peut définir le nb adapté pour chaque classe
# ros = RandomOverSampler(random_state=42, sampling_strategy={"managing_director_small_medium_company":100,
#                                                             "specialist": 100,
#                                                             "director_business_unit_leader": 100,
#                                                             "bereichsleiter": 1000})
# print(y_train.value_counts())
# manager_team_leader                       3470
# senior_specialist_or_project_manager      2138
# bereichsleiter                            1000
# specialist                                 100
# director_business_unit_leader              100
# managing_director_small_medium_company     100
# Name: career_level, dtype: int64

# Par contre, le point négatif de RandomOverSampler c'est qu'il ne génère que les mêmes données (donc pas variées).


# Pour avoir les nouvelles données plus variées, on va utiliser SMOTE (ou SMOTEN)
ros = SMOTEN(random_state=42,
             sampling_strategy={"managing_director_small_medium_company":100,
                                "specialist": 100,
                                "director_business_unit_leader": 100,
                                "bereichsleiter": 1000},
             k_neighbors=2)



### PIPELINE DE PREPROCESSING

preprocessor = ColumnTransformer(transformers=[
    ("title_features", TfidfVectorizer(), "title"),
    ("location_features", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("des_features", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function_features", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_features", TfidfVectorizer(), "industry"),
])



model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier()),
])


params = {
    "model__n_estimators": [50, 100, 200],
    "model__criterion": ["gini", "entropy", "log_loss"],
    "feature_selection__percentile": [5, 10, 20],
    "preprocessor__des_features__min_df": [0.01, 0.02, 0.05],
    "preprocessor__des_features__max_df": [0.90, 0.95, 0.99],
    "preprocessor__des_features__ngram_range": [(1, 1), (1, 2)],
}


grid_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    scoring="f1_weighted",
    cv=5,
    n_jobs=1,
    verbose=1,
    n_iter=50
)



grid_search.fit(X_train, y_train)


y_predict = grid_search.predict(X_test)

print(grid_search.best_params_)
print(grid_search.best_score_)


print(classification_report(y_test, y_predict))
#                                         precision    recall  f1-score   support

#                         bereichsleiter       0.62      0.04      0.08       192
#          director_business_unit_leader       1.00      0.29      0.44        14
#                    manager_team_leader       0.62      0.49      0.55       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.71      0.96      0.81       868
#                             specialist       0.00      0.00      0.00         6

#                               accuracy                           0.68      1615
#                              macro avg       0.49      0.30      0.31      1615 # pas pertinent pour un dataset imbalanced comme celui-ci
#                           weighted avg       0.67      0.68      0.63      1615



# avec df_min et df_max
#                                         precision    recall  f1-score   support

#                         bereichsleiter       0.67      0.05      0.10       192
#          director_business_unit_leader       1.00      0.29      0.44        14
#                    manager_team_leader       0.64      0.70      0.67       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.80      0.94      0.87       868
#                             specialist       0.00      0.00      0.00         6

#                               accuracy                           0.75      1615
#                              macro avg       0.52      0.33      0.35      1615
#                           weighted avg       0.73      0.75      0.70      1615



# avec RandomizedSearchCV
#                                         precision    recall  f1-score   support

#                         bereichsleiter       0.67      0.09      0.16       192
#          director_business_unit_leader       1.00      0.29      0.44        14
#                    manager_team_leader       0.63      0.73      0.68       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.83      0.92      0.87       868
#                             specialist       0.00      0.00      0.00         6

#                               accuracy                           0.75      1615
#                              macro avg       0.52      0.34      0.36      1615
#                           weighted avg       0.74      0.75      0.72      1615



