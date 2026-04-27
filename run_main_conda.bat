@echo off
setlocal

set "PYTHON_EXE=C:\Users\26409\.conda\envs\urban_renovation\python.exe"
set "SCRIPT_PATH=C:\Users\26409\Desktop\Urban Renovation\scripts\pipeline\main_py313.py"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python interpreter not found:
    echo %PYTHON_EXE%
    exit /b 1
)

if not exist "%SCRIPT_PATH%" (
    echo [ERROR] Entry script not found:
    echo %SCRIPT_PATH%
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_PATH%" %*
