@echo off
setlocal

cd /d "%~dp0"

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

REM Create logs directory if not exists
if not exist "logs" mkdir logs

REM Generate timestamp for log file
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set LOG_FILE=logs\daily_run_%mydate%_%mytime%.log

echo [%date% %time%] ========== Daily Rebalance Started ========== >> "%LOG_FILE%"

REM Run execution with logging
echo [Execution] Running daily rebalance...
"%PYTHON_EXE%" -m ancser_quant.execution.main_loop --run-once --force >> "%LOG_FILE%" 2>&1

if errorlevel 1 (
    echo [ERROR] Daily run failed. Check %LOG_FILE% for details.
    type "%LOG_FILE%"
) else (
    echo [SUCCESS] Daily run completed. Execution log: %LOG_FILE%
    echo.
    echo [Log Summary]
    findstr /I "generated\|submitted\|error\|account\|order" "%LOG_FILE%" || echo No trade details found
)

echo.
echo [%date% %time%] ========== Daily Rebalance Completed ========== >> "%LOG_FILE%"

REM Optional: Start Dashboard after rebalance
timeout /t 3 /nobreak
start "Ancser Dashboard" "%PYTHON_EXE%" -m streamlit run frontend/app.py

pause
endlocal
