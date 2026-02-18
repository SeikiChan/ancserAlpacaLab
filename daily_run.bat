@echo off
setlocal
cd /d "%~dp0"
<<<<<<< Updated upstream
call .venv\Scripts\activate.bat || echo "Virtualenv not found, using global python"
start "Ancser Dashboard" python -m streamlit run frontend/app.py
python -m ancser_quant.execution.main_loop --run-once
=======

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [Startup] Checking dependencies...
"%PYTHON_EXE%" -c "import pandas,polars,numpy,pyarrow,pytz,apscheduler,streamlit,plotly,alpaca,dotenv,yfinance" >nul 2>&1
if errorlevel 1 (
    echo [Startup] Missing dependencies detected. Installing requirements...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo [ERROR] Failed to upgrade pip.
        echo Please run setup.bat and try again.
        pause
        exit /b 1
    )

    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        echo Please run setup.bat and try again.
        pause
        exit /b 1
    )
)

start "Ancser Dashboard" "%PYTHON_EXE%" -m streamlit run dashboard/app.py
"%PYTHON_EXE%" -m ancser_quant.execution.main_loop --run-once
if errorlevel 1 (
    echo.
    echo [ERROR] Daily run failed. Check traceback above.
)

pause
endlocal
>>>>>>> Stashed changes
