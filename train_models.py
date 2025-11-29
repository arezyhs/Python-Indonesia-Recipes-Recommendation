"""
Main training script for the Indonesian Recipe Recommendation System
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_processor import DataProcessor
from src.models.recommendation_system import RecipeRecommendationSystem


def main():
    """Main training function."""
    print("=== Indonesian Recipe Recommendation System ===")
    print("Starting training process...")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Process data
    try:
        df = processor.process_all()
        print(f"Data processing completed. Total recipes: {len(df)}")
        
        # Print data summary
        summary = processor.get_data_summary()
        print("\nData Summary:")
        print(f"- Total recipes: {summary['total_recipes']}")
        print(f"- Categories: {summary['categories']}")
        print(f"- Average text length: {summary['avg_text_length']:.2f} characters")
        
    except Exception as e:
        print(f"Error in data processing: {e}")
        return
    
    # Initialize recommendation system
    rec_system = RecipeRecommendationSystem()
    
    # Train the system
    try:
        rec_system.fit(df)
        print("Model training completed!")
        
        # Print model info
        model_info = rec_system.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"- {key}: {value}")
        
        # Save models
        rec_system.save_models()
        
        print("\n=== Training Completed Successfully! ===")
        
        # Test the system with sample queries
        print("\n=== Testing System ===")
        test_queries = [
            "ayam goreng",
            "masakan tahu",
            "resep ikan"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            recommendations = rec_system.recommend_recipes(query, top_n=3)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']} (Score: {rec['similarity_score']:.4f})")
        
    except Exception as e:
        print(f"Error in model training: {e}")
        return


if __name__ == "__main__":
    main()