@echo off
setlocal enabledelayedexpansion
title Sakina FL Automation Tool

echo ========================================
echo   AVAILABLE STRATEGIES:
echo   FEDAVG, FEDPROX, FEDADAGRAD, FEDYOGI, FEDADAM
echo ========================================
set /p strat="Enter Strategy Name: "

echo Starting Server...
start "FL Server" cmd /k "python server.py %strat%"
timeout /t 5

echo Launching Clients (S2-S17, skipping S12)...
for /L %%i in (2,1,17) do (
    if %%i NEQ 12 (
        echo Starting Subject S%%i...
        start "Client S%%i" cmd /c "python client.py S%%i"
        timeout /t 1 > nil
    )
)
echo All components are active.
pause