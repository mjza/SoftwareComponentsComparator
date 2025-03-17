# SoftwareComponentComparator
A repository for comparing OSS components using AI 

## Creating the Virtual Environment
First, navigate to your project's root directory in your terminal. Then, create a virtual environment named venv (or another name of your choice) by running:

```
python -m venv venv
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

With the virtual environment activated, install the dependencies listed in `requirements.txt` by running:

```
pip install -r requirements.txt
```

## Configuration
Copy and paste the `.env.example` file and rename the new file to `.env`. 
Edit the `.env` file.

## Run
Run the `main.py` file to run the program. 
```bash
python main.py
```
