# ROBO-DOC – Multi-Disease Healthcare Chatbot

ROBO-DOC is a Flask-based healthcare chatbot that uses natural language processing (NLP), symptom matching, semantic similarity, and a pre-trained K-Nearest Neighbors (KNN) machine-learning model to identify possible diseases from user-provided symptoms.

> **Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 1. Features

- Interactive web-based healthcare chatbot
- Natural-language symptom input
- Symptom preprocessing using spaCy
- Lemmatization and text processing
- WordNet/Lesk-based semantic processing using NLTK
- Symptom similarity matching
- Pre-trained KNN disease prediction model
- Disease descriptions
- Symptom severity information
- Recommended precautions
- Flask-based backend and HTML/CSS frontend
- Conversation maintained using Flask sessions

---

## 2. Technologies Used

### Backend
- Python
- Flask
- Pandas
- NumPy
- Joblib
- scikit-learn

### NLP
- spaCy
- NLTK
- WordNet
- Lesk

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- K-Nearest Neighbors (KNN)

---

## 3. Project Structure

Make sure the extracted project has a structure similar to:

```text
Multidisease_Chatbot-main/
│
├── app1.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── knn.pkl
│   └── tfidfsymptoms.csv
│
├── Medical_dataset/
│   ├── Training.csv
│   ├── Testing.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   ├── symptom_severity.csv
│   └── ...
│
├── templates/
│   └── home.html
│
├── static/
│   └── styles/
│       └── style.css
│
├── screens/
└── Draft/
```

The most important files/folders for running the application are:

```text
app1.py
requirements.txt
model/
Medical_dataset/
templates/
static/
```

---

# 4. Requirements

Recommended:

- Python 3.10 or Python 3.11
- pip
- Virtual environment
- Internet connection for installing Python packages and the spaCy language model

The supplied `knn.pkl` model should be used with the compatible scikit-learn version specified in `requirements.txt`.

---

# 5. Installation on Windows

## Step 1 – Open the project folder

Open PowerShell or Command Prompt and navigate to the folder that contains:

```text
app1.py
requirements.txt
model
Medical_dataset
templates
static
```

For example:

```powershell
cd "path\to\Multidisease_Chatbot-main"
```

Do not copy the example path literally. Replace it with the location where you extracted the project.

Check that you are in the correct folder:

```powershell
dir
```

You should see:

```text
app1.py
requirements.txt
model
Medical_dataset
templates
static
```

---

# 6. Create a Virtual Environment

Run:

```powershell
python -m venv .venv
```

---

# 7. Activate the Virtual Environment

For PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of the terminal prompt.

Example:

```text
(.venv) PS ...\Multidisease_Chatbot-main>
```

---

# 8. Install Dependencies

First upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Then install the project requirements:

```powershell
python -m pip install -r requirements.txt
```

If the project does not already contain a complete `requirements.txt`, install the main dependencies manually:

```powershell
python -m pip install flask pandas numpy joblib scikit-learn spacy nltk
```

---

# 9. Install the spaCy Language Model

The chatbot uses the English spaCy model.

Run:

```powershell
python -m spacy download en_core_web_sm
```

Verify it:

```powershell
python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model is installed')"
```

---

# 10. Download NLTK Data

The chatbot uses NLTK resources for WordNet and related NLP processing.

Run:

```powershell
python -m nltk.downloader wordnet omw-1.4 punkt punkt_tab
```

If your NLTK version does not require `punkt_tab`, the command may simply report that the resource is already available or unavailable for that version.

---

# 11. Verify the Environment

Run:

```powershell
python -c "import flask, pandas, numpy, joblib, sklearn, spacy, nltk; print('All required packages are installed')"
```

You can also check the Python interpreter being used:

```powershell
python --version
```

and:

```powershell
python -m pip --version
```

The pip path should point to your `.venv` environment.

---

# 12. Run the Chatbot

Make sure you are in the directory containing `app1.py`.

Run:

```powershell
python app1.py
```

The Flask application should start and display a local address similar to:

```text
* Running on http://127.0.0.1:5000
```

Open a browser and visit:

```text
http://127.0.0.1:5000
```

---

# 13. Quick Start

After extracting the project, the complete Windows setup is:

```powershell
cd "path\to\Multidisease_Chatbot-main"

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -m spacy download en_core_web_sm

python -m nltk.downloader wordnet omw-1.4 punkt punkt_tab

python app1.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 14. How the Chatbot Works

The overall processing flow is:

```text
User
  |
  v
Chatbot Web Interface
  |
  v
Flask Application
  |
  v
User symptom text
  |
  v
Text preprocessing
  |
  +--> Tokenization
  |
  +--> Lemmatization
  |
  +--> Symptom matching
  |
  +--> Semantic similarity
  |
  v
Recognized symptoms
  |
  v
Feature vector
  |
  v
KNN model
  |
  v
Possible disease
  |
  +--> Disease description
  |
  +--> Severity information
  |
  +--> Precautions
  |
  v
Chatbot response
```

---

# 15. Machine Learning Model

The project uses a pre-trained KNN model stored at:

```text
model/knn.pkl
```

The model is loaded by the Flask application using Joblib.

The model works with the symptom feature representation generated from the project's medical dataset.

The symptom-related data is stored in:

```text
Medical_dataset/Training.csv
Medical_dataset/Testing.csv
```

The project also contains supporting files for:

```text
Disease descriptions
Symptom severity
Disease precautions
```

---

# 16. NLP Processing

The chatbot uses spaCy to process user input.

For example, the user may enter:

```text
I am having severe headache
```

The chatbot preprocesses the sentence and attempts to identify the corresponding symptom.

Lemmatization helps map different word forms to a common representation.

For example:

```text
headaches
```

may be normalized toward:

```text
headache
```

The project also uses NLTK WordNet/Lesk-based processing for semantic analysis.

---

# 17. Example Interaction

A typical interaction may look like:

```text
User:
Hello

Chatbot:
Welcome to ROBO-DOC.

User:
I have headache

Chatbot:
[Processes and matches the symptom]

User:
I also have nausea

Chatbot:
[Processes the additional symptom]

Chatbot:
Possible disease:
<Predicted disease>

Description:
<Disease description>

Precautions:
<Recommended precautions>
```

The exact conversation depends on the application's current chatbot logic and the symptoms available in the dataset.

---

# 18. Important: Use Symptoms From the Dataset

The chatbot does not understand every possible medical phrase.

Its prediction depends on the symptoms represented in the supplied dataset.

If the chatbot cannot recognize an entered symptom, try a simpler form.

For example:

```text
headache
fever
cough
vomiting
fatigue
nausea
```

instead of a long sentence containing many unrelated words.

---

# 19. Common Errors and Solutions

## Error 1 – `No module named spacy`

Run:

```powershell
python -m pip install spacy
```

Then:

```powershell
python -m spacy download en_core_web_sm
```

---

## Error 2 – `No module named nltk`

Run:

```powershell
python -m pip install nltk
```

Then:

```powershell
python -m nltk.downloader wordnet omw-1.4 punkt punkt_tab
```

---

## Error 3 – `No module named flask`

Run:

```powershell
python -m pip install flask
```

Or install everything:

```powershell
python -m pip install -r requirements.txt
```

---

## Error 4 – `No module named sklearn`

Run:

```powershell
python -m pip install scikit-learn
```

If the supplied model requires a specific version, install the version specified by `requirements.txt`.

---

## Error 5 – `Could not open requirements.txt`

This means that you are not in the project root.

Run:

```powershell
dir
```

You need to see:

```text
app1.py
requirements.txt
```

If you do not see them, navigate to the directory containing those files.

You can find the file with:

```powershell
Get-ChildItem -Recurse -Filter requirements.txt | Select-Object FullName
```

---

## Error 6 – `can't open file 'app1.py'`

You are in the wrong directory.

Find the file:

```powershell
Get-ChildItem -Recurse -Filter app1.py | Select-Object FullName
```

Then change to the folder containing `app1.py`:

```powershell
cd "path\to\folder"
```

Run:

```powershell
python app1.py
```

---

## Error 7 – `Can't find model 'en_core_web_sm'`

Run:

```powershell
python -m spacy download en_core_web_sm
```

Then verify:

```powershell
python -c "import spacy; print(spacy.load('en_core_web_sm'))"
```

---

## Error 8 – NLTK `Resource wordnet not found`

Run:

```powershell
python -m nltk.downloader wordnet omw-1.4
```

For tokenizer-related errors:

```powershell
python -m nltk.downloader punkt punkt_tab
```

---

## Error 9 – `invalid load key` / model loading error

This usually indicates that the model file is not a valid Joblib/Pickle file or that the file was created with an incompatible serialization/version.

First make sure you are using the supplied:

```text
model/knn.pkl
```

and that it has not been renamed or replaced.

Then verify scikit-learn:

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

Use the version specified in `requirements.txt`.

---

## Error 10 – Port 5000 already in use

If another Flask application is already running, stop it or change the port in `app1.py`.

For example:

```python
app.run(port=5001)
```

Then open:

```text
http://127.0.0.1:5001
```

---

## Error 11 – Chatbot does not recognize the symptom

Try entering the symptom using a simple dataset-style name.

For example:

```text
headache
```

instead of:

```text
I have been experiencing a really bad headache since yesterday
```

This project relies on the symptom vocabulary and matching logic implemented in the dataset/application.

---

# 20. Do I Need to Train the Model?

Normally, **no**.

The project already contains:

```text
model/knn.pkl
```

Therefore you can directly run:

```powershell
python app1.py
```

You only need to retrain the model if you intentionally want to modify the machine-learning pipeline or dataset.

---

# 21. Stopping the Application

In the terminal where Flask is running, press:

```text
Ctrl + C
```

---

# 22. Deactivate the Virtual Environment

After stopping the application:

```powershell
deactivate
```

---

# 23. Linux / macOS

Create the environment:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Install NLTK data:

```bash
python -m nltk.downloader wordnet omw-1.4 punkt punkt_tab
```

Run:

```bash
python app1.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 24. Development Notes

The application consists of:

```text
app1.py
```

for the Flask backend,

```text
templates/home.html
```

for the chatbot interface,

```text
static/styles/style.css
```

for styling,

```text
model/knn.pkl
```

for the trained KNN model,

and:

```text
Medical_dataset/
```

for the medical/symptom data.

Keep these folders in their original relative locations.

---

# 25. Medical Disclaimer

ROBO-DOC is an academic project demonstrating natural-language processing and machine-learning techniques.

The output of the application:

- may be incorrect,
- is based on the supplied dataset and trained model,
- does not constitute a medical diagnosis,
- should not be used to select or change treatment,
- should not replace consultation with a qualified healthcare professional.

For real medical concerns, seek appropriate professional medical care.

---

## 26. License / Usage

This project is intended for academic, educational, and demonstration purposes. Check the licenses and terms of the original datasets, libraries, and models before redistributing the project.
