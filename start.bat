@echo off
chcp 65001 >nul
title TrendSense Launcher

echo ================================================================
echo   TrendSense
echo ================================================================
echo.
echo   Usage: activate your Python env first, then double-click
echo     conda activate pro
echo     start.bat
echo.

set "PROJ_DIR=%~dp0"

:: ── 首次运行检查：缓存文件是否就绪 ──
if not exist "%PROJ_DIR%data\clean_transactions.parquet" (
    echo [WARN] 数据文件缺失，请先运行数据准备脚本:
    echo        python modules/preprocess.py
    echo        python modules/lstm_popularity.py
    echo        python modules/cf.py
    echo        python modules/inventory.py
    echo.
    choice /c yn /m "是否继续启动？(可能报错)"
    if errorlevel 2 exit /b
)

echo [Start] Flask AI backend (port 5000)
start "AI Chat" /d "%PROJ_DIR%" cmd /k "echo === AI Chat === && echo If ModuleNotFoundError: activate your Python env first && echo. && python ai_chat\api.py"

ping -n 3 127.0.0.1 >nul

echo [Start] Streamlit frontend
start "Streamlit" /d "%PROJ_DIR%" cmd /k "echo === Dashboard === && echo If ModuleNotFoundError: activate your Python env first && echo. && streamlit run app.py"

echo.
echo ================================================================
echo   Frontend: http://localhost:8501
echo   Stop: close both terminal windows
echo ================================================================
echo.
pause
