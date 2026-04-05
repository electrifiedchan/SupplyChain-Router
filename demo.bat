@echo off
color 0A
cls

echo.
echo  +--------------------------------------------------------------+
echo  ^|      SUPPLY CHAIN ROUTER - LIVE AI DEMO                     ^|
echo  ^|      Disaster Relief Logistics x NVIDIA NIM                 ^|
echo  +--------------------------------------------------------------+
echo.
echo  What you are about to see:
echo.
echo    An AI (Mistral Devstral-2, 123B parameters) is given a
echo    disaster scenario - helicopters, supply pallets, weight
echo    limits, and a hidden chemical/medical safety trap.
echo.
echo    It routes supplies across 3 difficulty levels:
echo      [EASY]   4 pallets, 2 helicopters, no traps
echo      [MEDIUM] 5 pallets, 2 helicopters, critical priority
echo      [HARD]   6 pallets, 3 helicopters, hazmat trap active
echo.
echo    The AI is running LIVE against our Hugging Face server.
echo    Every decision is a real API call to NVIDIA's cloud.
echo.
echo  Live server: https://electrifiedchan-disaster-relief-logistics.hf.space
echo.
pause

cls
echo.
echo  Connecting to Hugging Face Space...
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo  Run:  python -m venv venv   then:  pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python test_3modes.py

echo.
echo  +--------------------------------------------------------------+
echo  GitHub  : https://github.com/electrifiedchan/SupplyChain-Router
echo  HF Space: https://electrifiedchan-disaster-relief-logistics.hf.space
echo  OpenEnv Hackathon submission - April 2026
echo  +--------------------------------------------------------------+
echo.
pause
