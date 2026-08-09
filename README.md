# ROBO-DOC – Multi-Disease Healthcare Chatbot

ROBO-DOC is a Flask-based healthcare chatbot that uses natural language processing (NLP), symptom matching, semantic similarity, and a pre-trained K-Nearest Neighbors (KNN) machine-learning model to identify possible diseases from user-provided symptoms.

> **Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

## How to Run

From the folder containing `app1.py` and `requirements.txt`:

```powershell
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

The README includes the complete project structure and detailed troubleshooting for errors such as:

* `No module named spacy`
* `No module named nltk`
* `Could not open requirements.txt`
* `can't open file 'app1.py'`
* missing `en_core_web_sm`
* missing NLTK resources
* KNN model loading errors
* port 5000 already being used
* symptoms not being recognized

It also explains the NLP pipeline, KNN model, dataset, Flask architecture, and chatbot workflow.
