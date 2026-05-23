@echo off
REM =============================================================================
REM  build_exe.bat — Build a single-file Windows EXE from the OCR FastAPI app
REM
REM  BEFORE RUNNING THIS:
REM    Make sure you have run the app at least once on THIS machine so that
REM    EasyOCR has already downloaded its model files (~100 MB).
REM    Default location: C:\Users\<YOU>\.EasyOCR\model\
REM
REM  HOW TO USE (on Windows):
REM    1. Open Command Prompt in this project folder
REM    2. Run:  build_exe.bat
REM    3. Find the finished EXE at:  dist\OCRApp.exe
REM
REM  WHAT THE CLIENT GETS:
REM    - dist\OCRApp.exe   (fully self-contained, NO Python needed)
REM    - Double-click → browser opens → app runs
REM    - Creates rawdat.db and static\uploads\ next to the EXE on first run
REM    - Works 100% OFFLINE after bundling (models are inside the EXE)
REM =============================================================================

echo.
echo ============================================================
echo  Step 1: Install / upgrade PyInstaller
echo ============================================================
pip install --upgrade pyinstaller
if errorlevel 1 (
    echo ERROR: pip install failed. Is Python on your PATH?
    pause & exit /b 1
)

REM ── Locate the EasyOCR model cache ──────────────────────────────────────────
REM EasyOCR stores models in %USERPROFILE%\.EasyOCR\model\ by default.
REM We bundle them so the client can work OFFLINE.
set "EASYOCR_MODEL_DIR=%USERPROFILE%\.EasyOCR\model"

if not exist "%EASYOCR_MODEL_DIR%" (
    echo.
    echo WARNING: EasyOCR model folder not found at:
    echo   %EASYOCR_MODEL_DIR%
    echo.
    echo Run the app normally once first so EasyOCR downloads its models,
    echo then re-run this build script.
    echo.
    echo The EXE will still be built, but the client will need internet
    echo access on first launch to download models.
    echo.
    pause
)

echo.
echo ============================================================
echo  Step 2: Building EXE  (can take several minutes)
echo ============================================================

REM Build the base command
set PYINSTALLER_CMD=pyinstaller ^
    --onefile ^
    --name "OCRApp" ^
    --add-data "templates;templates" ^
    --add-data "static;static"

REM Bundle EasyOCR models if they exist (for offline use)
if exist "%EASYOCR_MODEL_DIR%" (
    echo Bundling EasyOCR models from: %EASYOCR_MODEL_DIR%
    set PYINSTALLER_CMD=%PYINSTALLER_CMD% ^
        --add-data "%EASYOCR_MODEL_DIR%;.EasyOCR\model"
)

REM All hidden imports
set PYINSTALLER_CMD=%PYINSTALLER_CMD% ^
    --hidden-import "uvicorn.logging" ^
    --hidden-import "uvicorn.loops" ^
    --hidden-import "uvicorn.loops.auto" ^
    --hidden-import "uvicorn.protocols" ^
    --hidden-import "uvicorn.protocols.http" ^
    --hidden-import "uvicorn.protocols.http.auto" ^
    --hidden-import "uvicorn.protocols.websockets" ^
    --hidden-import "uvicorn.protocols.websockets.auto" ^
    --hidden-import "uvicorn.lifespan" ^
    --hidden-import "uvicorn.lifespan.on" ^
    --hidden-import "uvicorn.lifespan.off" ^
    --hidden-import "fastapi" ^
    --hidden-import "fastapi.templating" ^
    --hidden-import "fastapi.staticfiles" ^
    --hidden-import "starlette" ^
    --hidden-import "starlette.routing" ^
    --hidden-import "jinja2" ^
    --hidden-import "sqlalchemy" ^
    --hidden-import "sqlalchemy.dialects.sqlite" ^
    --hidden-import "easyocr" ^
    --hidden-import "cv2" ^
    --hidden-import "torch" ^
    --hidden-import "torchvision" ^
    --hidden-import "numpy" ^
    --hidden-import "fitz" ^
    --hidden-import "PIL" ^
    --hidden-import "python_multipart" ^
    --hidden-import "multipart" ^
    --hidden-import "aiofiles" ^
    --hidden-import "api" ^
    --hidden-import "database" ^
    --hidden-import "main" ^
    --hidden-import "process_document" ^
    --collect-all "easyocr" ^
    run_app.py

%PYINSTALLER_CMD%

echo.
if exist "dist\OCRApp.exe" (
    echo ============================================================
    echo  SUCCESS!
    echo.
    echo  EXE:  dist\OCRApp.exe
    echo.
    echo  Send ONLY this one file to your client.
    echo  They double-click it — no Python, no install needed.
    echo.
    echo  NOTE: First launch takes 30-90 sec (EasyOCR model loading).
    echo  Subsequent launches are faster.
    echo ============================================================
) else (
    echo ============================================================
    echo  BUILD FAILED. Check the output above for errors.
    echo ============================================================
    exit /b 1
)
pause
