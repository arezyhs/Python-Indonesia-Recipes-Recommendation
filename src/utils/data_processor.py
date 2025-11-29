"""
Data preprocessing utilities for Indonesian Recipe Recommendation System
"""
import pandas as pd
import numpy as np
import os
from typing import List, Tuple
from config.settings import DATA_DIR, DATASET_FILES, RECIPE_CATEGORIES


class DataProcessor:
    """Class for processing and cleaning recipe data."""
    
    def __init__(self):
        self.df = None
        
    def load_datasets(self) -> pd.DataFrame:
        """
        Load and combine all dataset files.
        
        Returns:
            pd.DataFrame: Combined dataset
        """
        print("Loading datasets...")
        dataframes = []
        
        for filename in DATASET_FILES:
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"Loaded {filename}: {len(df)} recipes")
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not dataframes:
            raise ValueError("No dataset files found!")
            
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total recipes loaded: {len(combined_df)}")
        
        self.df = combined_df
        return combined_df
    
    def clean_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean and preprocess the recipe data.
        
        Args:
            df: DataFrame to clean (uses self.df if not provided)
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
            
        print("Cleaning data...")
        
        # Remove unnecessary columns if they exist
        columns_to_drop = [col for col in ['Loves', 'URL'] if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        # Fill NaN values with empty strings
        df['Ingredients'] = df['Ingredients'].fillna('')
        df['Steps'] = df['Steps'].fillna('')
        df['Title'] = df['Title'].fillna('')
        
        # Convert title to lowercase for consistency
        df['Title'] = df['Title'].str.lower()
        
        # Create combined text column for TF-IDF
        df['text'] = df['Title'] + ' ' + df['Ingredients'] + ' ' + df['Steps']
        df['text'] = df['text'].fillna('')
        
        # Remove rows with empty text
        df = df[df['text'].str.strip() != '']
        
        print(f"Data cleaned. Remaining recipes: {len(df)}")
        return df
    
    def categorize_recipe(self, title: str) -> str:
        """
        Categorize recipe based on keywords in title.
        
        Args:
            title: Recipe title
            
        Returns:
            str: Recipe category
        """
        title = title.lower()
        
        for keyword, category in RECIPE_CATEGORIES.items():
            if keyword in title:
                return category
        
        return 'Lainnya'
    
    def add_categories(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add category column to the dataframe.
        
        Args:
            df: DataFrame to categorize (uses self.df if not provided)
            
        Returns:
            pd.DataFrame: DataFrame with categories
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
            
        print("Adding categories...")
        df['Category'] = df['Title'].apply(self.categorize_recipe)
        
        # Print category distribution
        print("Category distribution:")
        print(df['Category'].value_counts())
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace special characters used as separators
        text = text.replace('--', ' ')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def process_all(self) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Returns:
            pd.DataFrame: Fully processed dataset
        """
        # Load datasets
        df = self.load_datasets()
        
        # Clean data
        df = self.clean_data(df)
        
        # Add categories
        df = self.add_categories(df)
        
        # Preprocess text column
        df['text_cleaned'] = df['text'].apply(self.preprocess_text)
        
        self.df = df
        return df
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the processed data.
        
        Returns:
            dict: Summary statistics
        """
        if self.df is None:
            return {}
        
        return {
            'total_recipes': len(self.df),
            'categories': self.df['Category'].value_counts().to_dict(),
            'avg_text_length': self.df['text_cleaned'].str.len().mean(),
            'columns': list(self.df.columns)
        }