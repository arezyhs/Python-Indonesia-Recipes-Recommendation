#!/bin/bash
# Launcher script for Indonesian Recipe Recommendation System (Linux/macOS)

echo "=============================================="
echo "Indonesian Recipe Recommendation System"
echo "=============================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Check if models exist
if [ ! -f "models/tfidf_vectorizer.pkl" ]; then
    echo "Training models for first time..."
    echo "This may take a few minutes..."
    python train_models.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to train models"
        exit 1
    fi
fi

# Launch Streamlit app
echo
echo "Starting Indonesian Recipe Recommendation System..."
echo "Open your browser and go to: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

streamlit run app/streamlit_app.py