"""
Configuration settings for the Indonesian Recipe Recommendation System
"""
import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Data files
DATASET_FILES = [
    'dataset-ayam.csv',
    'dataset-ikan.csv', 
    'dataset-kambing.csv',
    'dataset-sapi.csv',
    'dataset-tahu.csv',
    'dataset-telur.csv',
    'dataset-tempe.csv',
    'dataset-udang.csv'
]

# Model files
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')
RF_MODEL_PATH = os.path.join(MODELS_DIR, 'rf_model.pkl')
PROCESSED_DATA_PATH = os.path.join(MODELS_DIR, 'processed_recipes.pkl')

# Model parameters
TFIDF_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}

SVM_PARAMS = {
    'kernel': 'linear',
    'C': 1.0,
    'random_state': 42
}

RF_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': 20
}

# Categories for recipe classification
RECIPE_CATEGORIES = {
    'ayam': 'Ayam',
    'ikan': 'Ikan',
    'kambing': 'Kambing',
    'sapi': 'Sapi',
    'tahu': 'Tahu',
    'tempe': 'Tempe',
    'udang': 'Udang',
    'telur': 'Telur'
}

# App settings
DEFAULT_RECOMMENDATIONS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42