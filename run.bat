@echo off
setlocal

echo Travel Advisor Agent
echo ====================

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ and try again.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install / sync dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

:: Check .env exists
if not exist ".env" (
    echo.
    echo WARNING: .env file not found.
    echo Copying .env.example to .env ...
    copy .env.example .env >nul
    echo Please open .env and add your OPENAI_API_KEY and TAVILY_API_KEY, then re-run this script.
    pause
    exit /b 1
)

:: Check API keys are not placeholders
findstr /c:"your_openai_api_key_here" .env >nul 2>&1
if not errorlevel 1 (
    echo.
    echo WARNING: OPENAI_API_KEY in .env still has the placeholder value.
    echo Edit .env with your real API keys and re-run this script.
    pause
    exit /b 1
)

findstr /c:"your_tavily_api_key_here" .env >nul 2>&1
if not errorlevel 1 (
    echo.
    echo WARNING: TAVILY_API_KEY in .env still has the placeholder value.
    echo Edit .env with your real API keys and re-run this script.
    pause
    exit /b 1
)

:: Launch Streamlit
echo.
echo Starting Travel Advisor Agent...
echo Open http://localhost:8501 in your browser.
echo Press Ctrl+C to stop.
echo.
streamlit run app.py

endlocal
