@echo off
REM Launcher script for Indonesian Recipe Recommendation System (Windows)

echo ==============================================
echo Indonesian Recipe Recommendation System
echo ==============================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models\tfidf_vectorizer.pkl" (
    echo Training models for first time...
    echo This may take a few minutes...
    python train_models.py
    if errorlevel 1 (
        echo ERROR: Failed to train models
        pause
        exit /b 1
    )
)

REM Launch Streamlit app
echo.
echo Starting Indonesian Recipe Recommendation System...
echo Open your browser and go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app/streamlit_app.py

pause