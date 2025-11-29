"""
Recipe recommendation system using TF-IDF and machine learning models
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple, Dict, Any
import nltk
from nltk.corpus import stopwords

from config.settings import (
    TFIDF_PARAMS, SVM_PARAMS, RF_PARAMS, 
    TFIDF_MODEL_PATH, SVM_MODEL_PATH, RF_MODEL_PATH, PROCESSED_DATA_PATH,
    DEFAULT_RECOMMENDATIONS, TEST_SIZE, RANDOM_STATE
)


class RecipeRecommendationSystem:
    """Recipe recommendation system using TF-IDF and ML models."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.svm_model = None
        self.rf_model = None
        self.tfidf_matrix = None
        self.df = None
        self.cosine_sim_matrix = None
        
        # Download NLTK data if needed
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK resources."""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def fit(self, df: pd.DataFrame) -> 'RecipeRecommendationSystem':
        """
        Fit the recommendation system with data.
        
        Args:
            df: Processed DataFrame with recipes
            
        Returns:
            self: Fitted recommendation system
        """
        self.df = df.copy()
        
        # Setup TF-IDF vectorizer
        print("Setting up TF-IDF vectorizer...")
        try:
            # Try to get Indonesian stopwords
            indonesian_stopwords = set(stopwords.words('indonesian'))
        except:
            # Fallback to English stopwords
            indonesian_stopwords = set(stopwords.words('english'))
            print("Indonesian stopwords not available, using English stopwords")
        
        # Update TFIDF_PARAMS with Indonesian stopwords
        tfidf_params = TFIDF_PARAMS.copy()
        tfidf_params['stop_words'] = list(indonesian_stopwords)
        
        self.tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
        
        # Fit TF-IDF on the text data
        print("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text_cleaned'])
        
        # Compute cosine similarity matrix
        print("Computing cosine similarity matrix...")
        self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Train classification models
        self._train_classification_models()
        
        return self
    
    def _train_classification_models(self):
        """Train SVM and Random Forest classification models."""
        if self.df is None or self.tfidf_matrix is None:
            raise ValueError("System must be fitted first!")
        
        print("Training classification models...")
        
        # Prepare data for classification
        X = self.tfidf_matrix
        y = self.df['Category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Train SVM model
        print("Training SVM model...")
        self.svm_model = SVC(**SVM_PARAMS)
        self.svm_model.fit(X_train, y_train)
        
        # Evaluate SVM
        y_pred_svm = self.svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, y_pred_svm)
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(**RF_PARAMS)
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate Random Forest
        y_pred_rf = self.rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        
        # Store evaluation results
        self.evaluation_results = {
            'svm_accuracy': svm_accuracy,
            'rf_accuracy': rf_accuracy,
            'svm_report': classification_report(y_test, y_pred_svm, output_dict=True),
            'rf_report': classification_report(y_test, y_pred_rf, output_dict=True)
        }
    
    def recommend_recipes(self, query: str, top_n: int = DEFAULT_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Recommend recipes based on query using cosine similarity.
        
        Args:
            query: Search query
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended recipes with details
        """
        if self.tfidf_vectorizer is None or self.df is None:
            raise ValueError("System must be fitted first!")
        
        # Transform query to TF-IDF
        query_tfidf = self.tfidf_vectorizer.transform([query.lower()])
        
        # Compute cosine similarity with all recipes
        cosine_sim = cosine_similarity(query_tfidf, self.tfidf_matrix)
        
        # Get top N similar recipes
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for i in range(min(top_n, len(sim_scores))):
            idx, score = sim_scores[i]
            recipe = self.df.iloc[idx]
            
            recommendations.append({
                'title': recipe['Title'].title(),
                'category': recipe['Category'],
                'ingredients': self._format_ingredients(recipe['Ingredients']),
                'steps': self._format_steps(recipe['Steps']),
                'similarity_score': float(score),
                'recipe_index': idx
            })
        
        return recommendations
    
    def predict_category(self, query: str, model_type: str = 'svm') -> Tuple[str, float]:
        """
        Predict recipe category for a given query.
        
        Args:
            query: Recipe text to classify
            model_type: 'svm' or 'rf' for model selection
            
        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("System must be fitted first!")
        
        # Choose model
        if model_type == 'svm':
            model = self.svm_model
        elif model_type == 'rf':
            model = self.rf_model
        else:
            raise ValueError("model_type must be 'svm' or 'rf'")
        
        if model is None:
            raise ValueError(f"{model_type} model not trained!")
        
        # Transform query
        query_tfidf = self.tfidf_vectorizer.transform([query.lower()])
        
        # Predict
        prediction = model.predict(query_tfidf)[0]
        
        # Get confidence score (for SVM, use decision function)
        if model_type == 'svm':
            confidence = np.max(model.decision_function(query_tfidf))
        else:  # Random Forest
            confidence = np.max(model.predict_proba(query_tfidf))
        
        return prediction, float(confidence)
    
    def get_similar_recipes(self, recipe_index: int, top_n: int = DEFAULT_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """
        Get similar recipes to a given recipe using precomputed cosine similarity.
        
        Args:
            recipe_index: Index of the reference recipe
            top_n: Number of similar recipes to return
            
        Returns:
            List of similar recipes
        """
        if self.cosine_sim_matrix is None or self.df is None:
            raise ValueError("System must be fitted first!")
        
        if recipe_index >= len(self.df):
            raise ValueError("Recipe index out of range!")
        
        # Get similarity scores for the given recipe
        sim_scores = list(enumerate(self.cosine_sim_matrix[recipe_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the recipe itself (first item)
        sim_scores = sim_scores[1:top_n+1]
        
        similar_recipes = []
        for idx, score in sim_scores:
            recipe = self.df.iloc[idx]
            similar_recipes.append({
                'title': recipe['Title'].title(),
                'category': recipe['Category'],
                'ingredients': self._format_ingredients(recipe['Ingredients']),
                'steps': self._format_steps(recipe['Steps']),
                'similarity_score': float(score),
                'recipe_index': idx
            })
        
        return similar_recipes
    
    def _format_ingredients(self, ingredients: str) -> List[str]:
        """Format ingredients string into a list."""
        if pd.isna(ingredients) or ingredients == '':
            return []
        
        # Split by '--' and clean
        ingredients_list = [ing.strip() for ing in ingredients.split('--') if ing.strip()]
        return ingredients_list
    
    def _format_steps(self, steps: str) -> List[str]:
        """Format steps string into a list."""
        if pd.isna(steps) or steps == '':
            return []
        
        # Split by '--' and clean
        steps_list = [step.strip() for step in steps.split('--') if step.strip()]
        return steps_list
    
    def save_models(self):
        """Save trained models to disk."""
        print("Saving models...")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(TFIDF_MODEL_PATH), exist_ok=True)
        
        # Save TF-IDF vectorizer
        with open(TFIDF_MODEL_PATH, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save SVM model
        if self.svm_model is not None:
            with open(SVM_MODEL_PATH, 'wb') as f:
                pickle.dump(self.svm_model, f)
        
        # Save Random Forest model
        if self.rf_model is not None:
            with open(RF_MODEL_PATH, 'wb') as f:
                pickle.dump(self.rf_model, f)
        
        # Save processed data and additional info
        model_data = {
            'df': self.df,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim_matrix': self.cosine_sim_matrix,
            'evaluation_results': getattr(self, 'evaluation_results', {})
        }
        
        with open(PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load trained models from disk."""
        print("Loading models...")
        
        try:
            # Load TF-IDF vectorizer
            with open(TFIDF_MODEL_PATH, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            # Load SVM model
            if os.path.exists(SVM_MODEL_PATH):
                with open(SVM_MODEL_PATH, 'rb') as f:
                    self.svm_model = pickle.load(f)
            
            # Load Random Forest model
            if os.path.exists(RF_MODEL_PATH):
                with open(RF_MODEL_PATH, 'rb') as f:
                    self.rf_model = pickle.load(f)
            
            # Load processed data
            if os.path.exists(PROCESSED_DATA_PATH):
                with open(PROCESSED_DATA_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                    self.df = model_data['df']
                    self.tfidf_matrix = model_data['tfidf_matrix']
                    self.cosine_sim_matrix = model_data['cosine_sim_matrix']
                    self.evaluation_results = model_data.get('evaluation_results', {})
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained models."""
        info = {
            'tfidf_fitted': self.tfidf_vectorizer is not None,
            'svm_fitted': self.svm_model is not None,
            'rf_fitted': self.rf_model is not None,
            'data_loaded': self.df is not None,
            'total_recipes': len(self.df) if self.df is not None else 0,
            'evaluation_results': getattr(self, 'evaluation_results', {})
        }
        
        if self.tfidf_vectorizer is not None:
            info['tfidf_features'] = len(self.tfidf_vectorizer.get_feature_names_out())
        
        return info