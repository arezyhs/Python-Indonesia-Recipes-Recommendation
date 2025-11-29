"""
Utility functions for the Indonesian Recipe Recommendation System
"""
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_recipe_text(text: str) -> str:
    """
    Format recipe text for better display.
    
    Args:
        text: Raw recipe text
        
    Returns:
        str: Formatted text
    """
    if pd.isna(text) or text == '':
        return ""
    
    # Replace separators with proper formatting
    text = text.replace('--', '\n• ')
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Add bullet point to the beginning if not present
    if not text.startswith('• '):
        text = '• ' + text
    
    return text


def extract_ingredients_list(ingredients: str) -> List[str]:
    """
    Extract ingredients as a clean list.
    
    Args:
        ingredients: Raw ingredients string
        
    Returns:
        List[str]: Clean list of ingredients
    """
    if pd.isna(ingredients) or ingredients == '':
        return []
    
    # Split by '--' and clean each ingredient
    ingredients_list = []
    for ingredient in ingredients.split('--'):
        ingredient = ingredient.strip()
        if ingredient:
            # Clean up common formatting issues
            ingredient = re.sub(r'\s+', ' ', ingredient)
            ingredients_list.append(ingredient)
    
    return ingredients_list


def extract_cooking_steps(steps: str) -> List[str]:
    """
    Extract cooking steps as a clean list.
    
    Args:
        steps: Raw steps string
        
    Returns:
        List[str]: Clean list of cooking steps
    """
    if pd.isna(steps) or steps == '':
        return []
    
    # Split by '--' and clean each step
    steps_list = []
    for step in steps.split('--'):
        step = step.strip()
        if step:
            # Clean up common formatting issues
            step = re.sub(r'\s+', ' ', step)
            steps_list.append(step)
    
    return steps_list


def calculate_recipe_complexity(ingredients: List[str], steps: List[str]) -> Dict[str, Any]:
    """
    Calculate recipe complexity metrics.
    
    Args:
        ingredients: List of ingredients
        steps: List of cooking steps
        
    Returns:
        Dict with complexity metrics
    """
    num_ingredients = len(ingredients)
    num_steps = len(steps)
    
    # Calculate average step length
    avg_step_length = np.mean([len(step) for step in steps]) if steps else 0
    
    # Simple complexity scoring
    if num_ingredients <= 5 and num_steps <= 3:
        complexity = "Easy"
        score = 1
    elif num_ingredients <= 10 and num_steps <= 6:
        complexity = "Medium"
        score = 2
    else:
        complexity = "Hard"
        score = 3
    
    return {
        'num_ingredients': num_ingredients,
        'num_steps': num_steps,
        'avg_step_length': avg_step_length,
        'complexity_level': complexity,
        'complexity_score': score
    }


def extract_cooking_time(steps: List[str]) -> Dict[str, int]:
    """
    Extract estimated cooking time from steps.
    
    Args:
        steps: List of cooking steps
        
    Returns:
        Dict with time estimates
    """
    time_patterns = [
        r'(\d+)\s*menit',
        r'(\d+)\s*jam',
        r'(\d+)\s*detik',
        r'selama\s*(\d+)',
    ]
    
    total_minutes = 0
    found_times = []
    
    for step in steps:
        step_lower = step.lower()
        
        for pattern in time_patterns:
            matches = re.findall(pattern, step_lower)
            for match in matches:
                time_val = int(match)
                
                if 'jam' in step_lower:
                    time_val *= 60
                elif 'detik' in step_lower:
                    time_val = max(1, time_val // 60)  # Convert to minutes, min 1
                
                found_times.append(time_val)
                total_minutes += time_val
    
    # Estimate based on complexity if no explicit time found
    if total_minutes == 0:
        # Base estimate on number of steps
        estimated_per_step = 10  # minutes per step
        total_minutes = len(steps) * estimated_per_step
    
    return {
        'total_minutes': total_minutes,
        'hours': total_minutes // 60,
        'minutes': total_minutes % 60,
        'found_explicit_times': len(found_times) > 0
    }


def detect_dietary_preferences(ingredients: List[str], title: str = "") -> List[str]:
    """
    Detect dietary preferences based on ingredients and title.
    
    Args:
        ingredients: List of ingredients
        title: Recipe title
        
    Returns:
        List[str]: Dietary tags
    """
    dietary_tags = []
    
    # Combine all text for analysis
    all_text = " ".join(ingredients + [title]).lower()
    
    # Vegetarian indicators
    meat_indicators = ['ayam', 'ikan', 'daging', 'sapi', 'kambing', 'udang', 'cumi', 'kepiting']
    has_meat = any(indicator in all_text for indicator in meat_indicators)
    
    if not has_meat:
        dietary_tags.append('Vegetarian')
    
    # Vegan indicators (no animal products)
    animal_products = ['telur', 'susu', 'keju', 'mentega', 'santan']
    has_animal_products = any(product in all_text for product in animal_products)
    
    if not has_meat and not has_animal_products:
        dietary_tags.append('Vegan')
    
    # Seafood
    seafood_indicators = ['ikan', 'udang', 'cumi', 'kepiting', 'kerang']
    has_seafood = any(indicator in all_text for indicator in seafood_indicators)
    
    if has_seafood:
        dietary_tags.append('Seafood')
    
    # Spicy level detection
    spicy_indicators = ['cabe', 'cabai', 'pedas', 'rawit', 'lombok']
    spicy_count = sum(indicator in all_text for indicator in spicy_indicators)
    
    if spicy_count >= 2:
        dietary_tags.append('Very Spicy')
    elif spicy_count == 1:
        dietary_tags.append('Spicy')
    else:
        dietary_tags.append('Mild')
    
    return dietary_tags


def generate_recipe_summary(recipe_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive recipe summary.
    
    Args:
        recipe_data: Recipe data dictionary
        
    Returns:
        Dict with enhanced recipe information
    """
    # Extract basic info
    title = recipe_data.get('title', '')
    ingredients = recipe_data.get('ingredients', [])
    steps = recipe_data.get('steps', [])
    
    # Calculate metrics
    complexity = calculate_recipe_complexity(ingredients, steps)
    cooking_time = extract_cooking_time(steps)
    dietary_tags = detect_dietary_preferences(ingredients, title)
    
    # Enhanced summary
    summary = {
        **recipe_data,  # Include original data
        'complexity': complexity,
        'cooking_time': cooking_time,
        'dietary_tags': dietary_tags,
        'summary_text': f"Resep {title} dengan {complexity['num_ingredients']} bahan dan {complexity['num_steps']} langkah. "
                       f"Estimasi waktu memasak: {cooking_time['total_minutes']} menit. "
                       f"Tingkat kesulitan: {complexity['complexity_level']}."
    }
    
    return summary


def validate_recipe_data(recipe_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate recipe data for completeness and quality.
    
    Args:
        recipe_data: Recipe data to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required fields
    required_fields = ['title', 'ingredients', 'steps']
    for field in required_fields:
        if field not in recipe_data or not recipe_data[field]:
            issues.append(f"Missing or empty field: {field}")
    
    # Check data quality
    if 'ingredients' in recipe_data:
        ingredients = recipe_data['ingredients']
        if isinstance(ingredients, list) and len(ingredients) < 2:
            issues.append("Recipe should have at least 2 ingredients")
        elif isinstance(ingredients, str) and len(ingredients.split('--')) < 2:
            issues.append("Recipe should have at least 2 ingredients")
    
    if 'steps' in recipe_data:
        steps = recipe_data['steps']
        if isinstance(steps, list) and len(steps) < 2:
            issues.append("Recipe should have at least 2 cooking steps")
        elif isinstance(steps, str) and len(steps.split('--')) < 2:
            issues.append("Recipe should have at least 2 cooking steps")
    
    # Check title length
    if 'title' in recipe_data:
        title = recipe_data['title']
        if len(title) < 3:
            issues.append("Recipe title too short")
        elif len(title) > 100:
            issues.append("Recipe title too long")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def create_recipe_card_html(recipe_data: Dict[str, Any]) -> str:
    """
    Create HTML for a recipe card display.
    
    Args:
        recipe_data: Recipe data
        
    Returns:
        str: HTML string for recipe card
    """
    # Generate enhanced summary
    enhanced_recipe = generate_recipe_summary(recipe_data)
    
    # Extract info
    title = enhanced_recipe.get('title', 'Untitled Recipe')
    category = enhanced_recipe.get('category', 'Lainnya')
    complexity = enhanced_recipe.get('complexity', {})
    cooking_time = enhanced_recipe.get('cooking_time', {})
    dietary_tags = enhanced_recipe.get('dietary_tags', [])
    similarity_score = enhanced_recipe.get('similarity_score', 0)
    
    # Color coding for categories
    category_colors = {
        'Ayam': '#FF6B6B', 'Ikan': '#4ECDC4', 'Kambing': '#45B7D1',
        'Sapi': '#96CEB4', 'Tahu': '#FFEAA7', 'Tempe': '#DDA0DD',
        'Udang': '#98D8C8', 'Telur': '#F7DC6F', 'Lainnya': '#BDC3C7'
    }
    
    category_color = category_colors.get(category, '#BDC3C7')
    
    html = f"""
    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background: white;">
        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #333;">{title}</h3>
            <span style="background: {category_color}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                {category}
            </span>
        </div>
        
        <div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 14px; color: #666;">
            <span>⏱️ {cooking_time.get('total_minutes', 0)} menit</span>
            <span>🥘 {complexity.get('num_ingredients', 0)} bahan</span>
            <span>📝 {complexity.get('num_steps', 0)} langkah</span>
            <span>📊 {complexity.get('complexity_level', 'Unknown')}</span>
            <span>🎯 {similarity_score:.3f}</span>
        </div>
        
        <div style="margin-bottom: 10px;">
            {''.join([f'<span style="background: #e9ecef; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-right: 5px;">{tag}</span>' for tag in dietary_tags])}
        </div>
        
        <p style="color: #666; font-size: 14px; margin: 0;">
            {enhanced_recipe.get('summary_text', '')}
        </p>
    </div>
    """
    
    return html