@echo off
rem - first run: setup; then opens the launcher
rem - run.bat --notebook: run the notebook in this console
rem - run.bat --self-test: status only, no window
rem - never deletes a .venv it did not create
setlocal

rem - go to the project folder
cd /d "%~dp0"
if errorlevel 1 goto err_folder

rem - FRESH=1: this run made .venv (only then may it be removed)
set "FRESH="

rem - marker file = setup done
if not exist ".venv\Scripts\python.exe" goto no_python_in_venv
if exist ".venv\setup_done.txt" goto have_venv

rem - no marker: accept only if libraries exist, then write the marker
".venv\Scripts\python.exe" -c "import pandas, sklearn, nbconvert" >nul 2>&1
if errorlevel 1 goto err_incomplete
>".venv\setup_done.txt" echo setup finished
goto have_venv

:no_python_in_venv
rem - .venv without Python: never touched
if exist ".venv" goto err_incomplete

rem - first time: find Python, make .venv, install
set "FRESH=1"
set "PYCMD="
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3.12"
if defined PYCMD goto make_venv
py -3 --version >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3"
if defined PYCMD goto make_venv
python --version >nul 2>&1
if not errorlevel 1 set "PYCMD=python"
if defined PYCMD goto make_venv
goto err_nopython

:make_venv
echo First-time setup: creating the virtual environment .venv ...
%PYCMD% -m venv .venv
if errorlevel 1 goto err_venv
echo Installing the libraries from requirements.txt (needs internet, may take a few minutes) ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto err_pip
>".venv\setup_done.txt" echo setup finished
echo Setup finished.

:have_venv
if /i "%~1"=="--notebook" goto run_notebook

rem - launcher; extra arguments passed on
".venv\Scripts\python.exe" launcher.py %*
if errorlevel 1 goto err_launcher
goto done

:run_notebook
if not exist "train.csv" goto err_train
if "%LAUNCHER_DRY_RUN%"=="1" goto dry_run
echo Running the whole notebook. This adds an entry to results\final_test_log.json.
".venv\Scripts\python.exe" -m jupyter nbconvert --to notebook --execute --inplace store_demand_forecast.ipynb
if errorlevel 1 goto err_notebook
echo.
echo Notebook finished successfully.
goto done_pause

:dry_run
".venv\Scripts\python.exe" -c "print('dry run')"
if errorlevel 1 goto err_notebook
goto done

:err_folder
echo ERROR: could not open the project folder.
goto fail

:err_nopython
echo ERROR: Python was not found. Install Python 3.12 from https://www.python.org/downloads/
echo        (tick "Add python.exe to PATH" in the installer), then double-click run.bat again.
goto fail

:err_venv
echo ERROR: could not create the virtual environment .venv.
if defined FRESH if exist ".venv" rmdir /s /q ".venv"
goto fail

:err_pip
echo ERROR: installing the libraries failed (see the messages above, often no internet).
echo        The unfinished .venv (created by this run) is removed so the next run starts clean.
if defined FRESH rmdir /s /q ".venv"
goto fail

:err_incomplete
echo ERROR: the .venv folder in this project looks incomplete (an earlier setup may
echo        have been interrupted, or the libraries are missing). It was NOT changed.
echo        Please delete the folder named .venv in this project yourself,
echo        then double-click run.bat again to set it up from scratch.
goto fail

:err_train
echo ERROR: train.csv is missing. Download it from
echo        https://www.kaggle.com/c/demand-forecasting-kernels-only (Data tab)
echo        and put it in this folder, next to run.bat.
goto fail

:err_notebook
echo ERROR: the notebook run did not finish (see the messages above).
goto fail

:err_launcher
echo ERROR: the launcher stopped with an error (see the messages above).
goto fail

:fail
echo.
pause
exit /b 1

:done_pause
echo.
pause
exit /b 0

:done
exit /b 0
