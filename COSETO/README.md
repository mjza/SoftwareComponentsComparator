# COSETO
A COmponent SElector TOol

# How to run the program

## Creating the Virtual Environment
First, navigate to your project's root directory in your terminal. Then, create a virtual environment named venv (or another name of your choice) by running:

```
python3 -m venv venv
```

This command creates a new directory named venv in your project directory, which contains a copy of the Python interpreter, the standard library, and various supporting files.

## Activating the Virtual Environment
Before you can start installing packages, you need to activate the virtual environment. 
Activation will ensure that the Python interpreter and tools within the virtual environment are used in preference to the system-wide Python installation.

1. **On macOS and Linux:**

```
source venv/bin/activate
```

2. **On Windows (cmd.exe):**

```
.\venv\Scripts\activate.bat
```

3. **On Windows (PowerShell) or VSC Terminal:**

```
.\venv\Scripts\Activate.ps1
```

Once activated, your terminal prompt must change to indicate that the virtual environment is active.

## Installing Dependencies

If you want to install all requirements at once use the following instruction with the virtual environment activated:

```bash
pip install -r requirements.txt
```

## Environment Variables
Make a copy of the `template.env` and rename it to `.env`, then copy and paste your provided database secret keys into the file.


## Start/Run the App
Use the following command to run the application:
```
python3 data_preprocess.py
python3 best_nli.py
python3 nli_classifier.py
python3 
```

This will open the github page in your default browser.
