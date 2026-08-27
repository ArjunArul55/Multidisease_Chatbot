import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import csv
import json
import itertools
import os
import re
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib
from flask import Flask, render_template, request, session
from google import genai

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-this")

nlp = spacy.load('en_core_web_sm')

# ==============================
# GEMINI AI CONFIGURATION
# ==============================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print("Gemini initialization error:", e)
else:
    print("WARNING: GEMINI_API_KEY is not configured. AI fallback features will be disabled.")


# save data (only initialize the file if it doesn't already exist,
# so restarting the app doesn't wipe out previously collected records)
def _init_data_file(filename='DATA.json'):
    if not os.path.exists(filename):
        with open(filename, 'w') as outfile:
            json.dump({"users": []}, outfile)


_init_data_file()


def write_json(new_data, filename='DATA.json'):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["users"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)


df_tr = pd.read_csv('Medical_dataset/Training.csv')
df_tt = pd.read_csv('Medical_dataset/Testing.csv')

symp = []
disease = []
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list())
    disease.append(df_tr.iloc[i, -1])

# I- GET ALL SYMPTOMS

all_symp_col = list(df_tr.columns[:-1])


def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace(
        'yellowing', 'yellow')


all_symp = [clean_symp(sym) for sym in (all_symp_col)]


def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)


all_symp_pr = [preprocess(sym) for sym in all_symp]

# associate each processed symp with column name
col_dict = dict(zip(all_symp_pr, all_symp_col))


# ==============================
# GEMINI HELPER FUNCTIONS
# ==============================

def ask_gemini(prompt):
    """
    Send a prompt to Gemini.
    Returns None if the API is unavailable or errors out.
    """
    if gemini_client is None:
        return None

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        print("Gemini API error:", e)

    return None


def classify_intent(message):
    """
    Determine what the user is trying to do.
    """
    prompt = f"""
You are the intent classifier for a medical symptom chatbot.

Classify the user's message into EXACTLY ONE of these categories:

greeting
medical
yes
no
quit
conversation
irrelevant

Definitions:

greeting:
hello, hi, hey, good morning, good evening, etc.

medical:
The user describes symptoms, pain, illness, disease,
health problems, or asks about their health.

yes:
yes, yeah, yep, sure, correct, okay, etc.

no:
no, nope, not really, etc.

quit:
bye, exit, quit, stop, end the conversation.

conversation:
Questions about this chatbot, its prediction,
how it works, or the current medical conversation.

irrelevant:
Questions unrelated to this medical assistant.

User message:
{message}

Return ONLY one category.
"""

    result = ask_gemini(prompt)

    if not result:
        # If Gemini is unavailable, use the local symptom matcher as
        # the fallback so real medical messages still enter the
        # prediction pipeline.
        try:
            if local_symptom_fallback(message):
                return "medical"
        except Exception:
            pass

        value = message.lower().strip()

        if value in {"hi", "hello", "hey", "good morning", "good evening"}:
            return "greeting"

        if value in {"yes", "yeah", "yep", "yup", "sure", "ok", "okay"}:
            return "yes"

        if value in {"no", "nope", "nah"}:
            return "no"

        if value in {"bye", "quit", "exit", "stop", "end"}:
            return "quit"

        return "conversation"

    result = result.lower().strip()

    allowed = {
        "greeting",
        "medical",
        "yes",
        "no",
        "quit",
        "conversation",
        "irrelevant"
    }

    if result in allowed:
        return result

    return "conversation"


def extract_symptoms_from_message(message):
    """
    Use Gemini to extract symptoms from natural language and map them to
    the exact symptom column names used by the existing ML model.
    """
    if not message:
        return []

    # Human-readable -> exact dataset column mapping.
    readable_to_column = {}
    for symptom in all_symp_col:
        readable = clean_symp(symptom).lower().strip()
        readable_to_column[readable] = symptom

    supported = list(readable_to_column.keys())

    prompt = f"""
You are the symptom-extraction component of a medical symptom prediction
application.

User message:
"{message}"

Supported symptoms in the machine-learning dataset:
{", ".join(supported)}

Task:
- Identify only symptoms that the user actually describes.
- Understand natural language, synonyms, spelling variations, and phrases.
- Map the user's words to the closest supported symptom.
- Do not diagnose a disease.
- Do not invent symptoms.
- Do not infer a symptom merely because it is commonly associated with
  another symptom.
- Return JSON only.

Format:
{{"symptoms":["supported symptom name"]}}
"""

    result = ask_gemini(prompt)

    if not result:
        return []

    try:
        cleaned = result.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        raw_symptoms = data.get("symptoms", [])

        if not isinstance(raw_symptoms, list):
            return []

        extracted = []

        # First: exact human-readable match.
        for item in raw_symptoms:
            if not isinstance(item, str):
                continue

            candidate = item.lower().strip()

            if candidate in readable_to_column:
                col = readable_to_column[candidate]
                if col not in extracted:
                    extracted.append(col)
                continue

            # Second: normalized comparison.
            candidate_processed = preprocess(candidate)
            for readable, col in readable_to_column.items():
                if candidate_processed == preprocess(readable):
                    if col not in extracted:
                        extracted.append(col)
                    break

        print("Gemini extracted symptoms:", raw_symptoms)
        print("Mapped dataset symptoms:", extracted)
        return extracted

    except Exception as e:
        print("Symptom extraction error:", e)
        print("Gemini raw response:", result)
        return []


def local_symptom_fallback(message):
    """
    Local fallback when Gemini is unavailable or returns no valid symptom.
    Uses exact/normalized substring matching and fuzzy similarity.
    """
    if not message:
        return []

    import difflib

    text = clean_symp(message).lower()
    processed_text = preprocess(text)
    matches = []

    # Exact human-readable phrase match.
    for symptom in all_symp_col:
        readable = clean_symp(symptom).lower().strip()
        if readable and readable in text:
            matches.append(symptom)

    # Normalized phrase match.
    for symptom in all_symp_col:
        readable = clean_symp(symptom).lower().strip()
        processed = preprocess(readable)

        if processed and processed in processed_text:
            matches.append(symptom)

    # Fuzzy match for short natural phrases.
    words = text.split()
    for symptom in all_symp_col:
        readable = clean_symp(symptom).lower().strip()
        if not readable:
            continue

        score = difflib.SequenceMatcher(None, text, readable).ratio()

        if score >= 0.82:
            matches.append(symptom)

        # Also compare individual n-grams for phrases such as
        # "pain in my stomach" -> "stomach pain".
        symptom_words = readable.split()
        n = len(symptom_words)

        if n > 0 and len(words) >= n:
            for i in range(len(words) - n + 1):
                chunk = " ".join(words[i:i+n])
                if difflib.SequenceMatcher(None, chunk, readable).ratio() >= 0.88:
                    matches.append(symptom)
                    break

    # Preserve dataset order and remove duplicates.
    unique = []
    for symptom in all_symp_col:
        if symptom in matches and symptom not in unique:
            unique.append(symptom)

    print("Local symptom fallback:", unique)
    return unique

def generate_ai_response(disease, symptoms):
    """
    Generate a natural-language explanation of the ML prediction.
    """
    description = description_list.get(
        disease,
        "No detailed description is available."
    )

    precautions = precautionDictionary.get(
        disease,
        []
    )

    prompt = f"""
You are the explanation assistant for a medical prediction chatbot.

The machine-learning model predicted:

{disease}

Symptoms provided:
{", ".join(symptoms)}

Disease description:
{description}

Precautions:
{", ".join(precautions)}

Give a concise and friendly explanation.

Rules:
- Do NOT say the user definitely has the disease.
- Clearly say this is an ML-based prediction, not a confirmed diagnosis.
- Explain the prediction in simple language.
- Include useful precautions from the provided list.
- Do not invent medication.
- Do not invent symptoms.
- Recommend consulting a qualified healthcare professional.
"""

    response = ask_gemini(prompt)

    if response:
        return response

    return (
        f"The machine-learning model predicted {disease}. "
        "This is not a confirmed medical diagnosis. "
        "Please consult a qualified healthcare professional."
    )


def ai_fallback_response(message, context_note=""):
    """
    Generic AI-powered response used whenever the deterministic
    symptom-matching pipeline (syntactic + semantic similarity, KNN)
    cannot map the user's message to anything in the dataset.
    """
    prompt = f"""
You are a helpful assistant embedded inside a medical symptom-checker chatbot.

The user's message could not be matched to any symptom in the
project's machine-learning dataset.

{context_note}

User message:
"{message}"

Reply briefly and helpfully, in 2-3 sentences:
- If it's a greeting, greet them back and invite them to describe a symptom.
- If it's a question about how this chatbot works, explain briefly that it is an
  ML-based symptom checker that predicts possible conditions from reported symptoms.
- If it's unrelated to health or symptoms, politely say you can only help with
  symptom-based predictions and ask them to describe a health concern.
- Never diagnose. Never invent symptoms. Never claim certainty.
"""
    response = ask_gemini(prompt)
    if response:
        return response

    return "I couldn't quite understand that. Could you describe a symptom you're experiencing?"


# II- Syntactic Similarity

def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a


def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])


def DoesExist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        for sym in permutations(comb):
            if sym in all_symp_pr:
                return sym
    return False


def jaccard_set(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(symp_t):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None


def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None


# III- Semantic Similarity

from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


def WSD(word, context):
    sens = lesk(context, word)
    return sens


def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))


def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)


def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a


def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis


def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()


# IV- Prediction Model (KNN)
knn_clf = joblib.load('model/knn.pkl')

# VI- SEVERITY / DESCRIPTION / PRECAUTION

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()


def getDescription():
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except Exception:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


getSeverityDict()
getprecautionDict()
getDescription()


def calc_condition(exp, days):
    total = 0
    for item in exp:
        if item in severityDictionary.keys():
            total = total + severityDictionary[item]
    if ((total * days) / (len(exp)) > 13):
        return 1
    else:
        return 0


def related_sym(psym1):
    if not psym1:
        return 0
    s = "could you be more specific, <br>"
    for num, it in enumerate(psym1):
        s += str(num) + ") " + clean_symp(it) + "<br>"
    s += "Select the one you meant."
    return s


# ============================================================
# ROBUST CHATBOT CONVERSATION ENGINE
# ============================================================

def normalize_yes_no(message):
    """Return True/False/None for natural-language yes/no answers."""
    if not message:
        return None

    value = message.lower().strip()

    yes_words = {
        "yes", "yeah", "yep", "yup", "sure", "correct", "okay", "ok",
        "true", "i do", "i have", "that's right", "that is right"
    }

    no_words = {
        "no", "nope", "nah", "not", "not really", "false", "i don't",
        "i do not", "i haven't", "i have not"
    }

    if value in yes_words:
        return True

    if value in no_words:
        return False

    # Handle natural variants such as "yes, I do" or "no, I don't".
    if value.startswith(("yes ", "yeah ", "yep ", "sure ")):
        return True

    if value.startswith(("no ", "nope ", "nah ")):
        return False

    return None


def parse_age(message):
    """Extract an age safely. Gemini is used first, regex is the fallback."""
    if not message:
        return None

    match = re.search(r"\b(\d{1,3})\b", message)
    if match:
        age = int(match.group(1))
        if 1 <= age <= 120:
            return age

    result = ask_gemini(f"""
Extract the person's age from this message:

"{message}"

Return ONLY the integer age.
If no valid age is present, return 0.
""")

    if result:
        match = re.search(r"\b(\d{1,3})\b", result)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 120:
                return age

    return None


def is_quit(message):
    if not message:
        return False

    return message.lower().strip() in {
        "q", "quit", "exit", "bye", "stop", "end", "close"
    }


def reset_diagnostic_keep_profile():
    """Keep profile information and start another assessment."""
    name = session.get("name", "User")
    age = session.get("age")
    gender = session.get("gender")

    session.clear()
    session["name"] = name
    session["age"] = age
    session["gender"] = gender
    session["step"] = "FS"
    session["all"] = []
    session["asked"] = []
    session["diseases"] = []

    return name


def start_new_assessment():
    session["all"] = []
    session["asked"] = []
    session["diseases"] = []
    session.pop("dis", None)
    session.pop("testpred", None)
    session.pop("symv", None)
    session.pop("disease", None)
    session["step"] = "FS"


def get_candidate_diseases(symptoms):
    """Return candidate diseases, safely."""
    if not symptoms:
        return []

    try:
        return possible_diseases(symptoms)
    except Exception as e:
        print("Disease filtering error:", e)
        return []


def add_symptoms(symptoms):
    """Add valid dataset symptoms to session without duplicates."""
    current = session.get("all", [])

    for symptom in symptoms:
        if symptom in all_symp_col and symptom not in current:
            current.append(symptom)

    session["all"] = current


def find_next_disease_question():
    """
    Find an unasked symptom belonging to the current candidate disease.
    """
    diseases = session.get("diseases", [])
    current = session.get("all", [])
    asked = session.get("asked", [])

    if not diseases:
        return None

    # Start with the first candidate disease.
    current_disease = diseases[0]
    session["dis"] = current_disease

    disease_symptoms = symVONdisease(df_tr, current_disease)

    for symptom in disease_symptoms:
        if symptom not in current and symptom not in asked:
            asked.append(symptom)
            session["asked"] = asked
            return symptom

    return None


def predict_current_symptoms():
    """
    Run the existing KNN model using the exact feature order expected by it.
    """
    symptoms = session.get("all", [])

    if not symptoms:
        return None

    try:
        vector = OHV(symptoms, all_symp_col)
        prediction = knn_clf.predict(vector)
        return prediction[0] if len(prediction) else None
    except Exception as e:
        print("KNN prediction error:", e)
        return None


def build_final_prediction_response(disease_name):
    """Generate a safe, friendly prediction message."""
    symptoms = session.get("all", [])

    response = generate_ai_response(
        disease_name,
        symptoms
    )

    if response:
        return response

    return (
        "The machine-learning model predicts <b>" +
        str(disease_name) +
        "</b> based on the symptoms you provided. "
        "This is not a confirmed medical diagnosis."
    )


def save_assessment():
    """
    Save only the existing application's assessment record.
    For production, replace DATA.json with a proper database.
    """
    try:
        y = {
            "Name": session.get("name", "User"),
            "Age": session.get("age"),
            "Gender": session.get("gender"),
            "Disease": session.get("disease"),
            "Sympts": session.get("all", [])
        }
        write_json(y)
    except Exception as e:
        print("Could not save assessment:", e)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    s = request.args.get("msg", "").strip()

    if not s:
        return "Please enter a message."

    # --------------------------------------------------------
    # GLOBAL QUIT HANDLING
    # --------------------------------------------------------
    if is_quit(s):
        name = session.get("name", "User")
        session.clear()
        return (
            "Thank you, " + str(name) +
            ", for using the medical assistant. "
            "Please consult a qualified healthcare professional "
            "for diagnosis or treatment."
        )

    # --------------------------------------------------------
    # INITIAL / NO SESSION
    # --------------------------------------------------------
    if "step" not in session:

        # Preserve the original application's OK -> name flow.
        if s.upper() == "OK":
            session.clear()
            session["step"] = "name"
            return "What is your name?"

        intent = classify_intent(s)

        if intent == "greeting":
            return (
                "Hello! 👋 I'm Medibot. "
                "I can help analyze symptoms using a machine-learning "
                "model. Tell me what symptoms you're experiencing."
            )

        if intent == "quit":
            session.clear()
            return "Alright, take care!"

        # Always try symptom extraction before rejecting a message as
        # conversation/irrelevant. This makes natural language robust.
        extracted = extract_symptoms_from_message(s)

        if not extracted:
            extracted = local_symptom_fallback(s)

        if extracted:
            add_symptoms(extracted)
            session["name"] = "User"
            session["step"] = "age"

            readable = ", ".join(
                clean_symp(x) for x in extracted
            )

            return (
                "I detected these symptoms: <b>" +
                readable +
                "</b>.<br>How old are you?"
            )

        if intent == "irrelevant":
            return (
                "I'm designed for health-related symptom assessment. "
                "I can't help with unrelated topics. "
                "Please describe a health symptom or concern."
            )

        if intent == "conversation":
            return ai_fallback_response(
                s,
                "The user has not started an assessment yet."
            )

        # If it is neither medical nor a known command, treat it as a name.
        session["name"] = s
        session["step"] = "age"
        return "How old are you?"

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------
    if session["step"] == "name":
        if len(s) < 1:
            return "Please enter your name."

        session["name"] = s
        session["step"] = "age"
        return "How old are you?"

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------
    if session["step"] == "age":
        age = parse_age(s)

        if age is None:
            return "Please enter your age, for example: 21."

        session["age"] = age
        session["step"] = "gender"

        return "Can you specify your gender?"

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------
    if session["step"] == "gender":
        gender = s.strip()

        if len(gender) < 1:
            return "Please specify your gender."

        session["gender"] = gender
        session["step"] = "FS"

        # If symptoms were already supplied before age/gender,
        # go directly to prediction flow.
        if session.get("all"):
            return (
                "Thank you. I have your information. "
                "Let's continue with your symptom assessment."
            )

        return (
            "Thank you, Mr/Ms " + str(session["name"]) +
            ". Please describe your main symptom."
        )

    # --------------------------------------------------------
    # FIRST SYMPTOM
    # --------------------------------------------------------
    if session["step"] == "FS":

        intent = classify_intent(s)

        if intent == "quit":
            session.clear()
            return "Alright, take care!"

        # Try the actual symptom extractor first. This prevents a
        # classification mistake from blocking a valid symptom.
        symptoms = extract_symptoms_from_message(s)

        if not symptoms:
            symptoms = local_symptom_fallback(s)

        if not symptoms and intent in ("greeting", "irrelevant", "conversation"):
            return ai_fallback_response(
                s,
                "The user is expected to describe a medical symptom."
            )

        if not symptoms:
            return (
                "I couldn't identify the symptom. "
                "Please describe it in another way, for example "
                "\"I have stomach pain\" or \"my head hurts\"."
            )

        add_symptoms(symptoms)
        session["step"] = "SS"

        readable = ", ".join(clean_symp(x) for x in symptoms)

        return (
            "I understood: <b>" + readable +
            "</b>. Do you have any other symptoms? "
            "You can say yes and describe them, or no."
        )

    # --------------------------------------------------------
    # SECOND / ADDITIONAL SYMPTOMS
    # --------------------------------------------------------
    if session["step"] == "SS":

        yn = normalize_yes_no(s)

        if yn is False:
            session["step"] = "PD"

        elif yn is True:
            session["step"] = "MORE_SYMPTOMS"
            return "Please describe your other symptom(s)."

        else:
            # Treat any medical-looking message as another symptom.
            intent = classify_intent(s)

            if intent in ("greeting", "irrelevant", "conversation"):
                return ai_fallback_response(
                    s,
                    "The user was asked whether they have another symptom."
                )

            if intent == "medical":
                extracted = extract_symptoms_from_message(s)

                if not extracted:
                    extracted = local_symptom_fallback(s)

                if extracted:
                    add_symptoms(extracted)
                    session["step"] = "PD"

                    return (
                        "I added: <b>" +
                        ", ".join(clean_symp(x) for x in extracted) +
                        "</b>. Let me analyze your symptoms."
                    )

            return (
                "Please say <b>yes</b> and describe another symptom, "
                "or say <b>no</b> if you don't have another symptom."
            )

        # Continue to disease filtering.
        return get_bot_response()

    # --------------------------------------------------------
    # MORE SYMPTOMS
    # --------------------------------------------------------
    if session["step"] == "MORE_SYMPTOMS":

        extracted = extract_symptoms_from_message(s)

        if not extracted:
            extracted = local_symptom_fallback(s)

        if not extracted:
            return (
                "I couldn't identify that symptom. "
                "Please describe it in another way."
            )

        add_symptoms(extracted)
        session["step"] = "PD"

        return (
            "I added: <b>" +
            ", ".join(clean_symp(x) for x in extracted) +
            "</b>. Let me analyze your symptoms."
        )

    # --------------------------------------------------------
    # FIND CANDIDATE DISEASES
    # --------------------------------------------------------
    if session["step"] == "PD":

        symptoms = session.get("all", [])
        diseases = get_candidate_diseases(symptoms)

        session["diseases"] = diseases
        session["asked"] = []

        if not diseases:
            session["step"] = "MORE_IF_NO_MATCH"

            return (
                "I couldn't find a reliable match from the symptoms "
                "provided. Please describe another symptom so I can "
                "narrow the possibilities."
            )

        session["step"] = "DIS"
        return get_bot_response()

    # --------------------------------------------------------
    # ASK DISEASE-SPECIFIC QUESTIONS
    # --------------------------------------------------------
    if session["step"] == "DIS":

        # If we arrived here after an earlier question, process yes/no.
        pending = session.get("pending_symptom")

        if pending:
            yn = normalize_yes_no(s)

            if yn is True:
                add_symptoms([pending])

            elif yn is None:
                # Don't consume an unrelated message as an answer.
                intent = classify_intent(s)

                if intent == "medical":
                    extracted = extract_symptoms_from_message(s)

                    if not extracted:
                        extracted = local_symptom_fallback(s)

                    if extracted:
                        add_symptoms(extracted)
                    else:
                        return (
                            "Please answer yes or no, or describe the "
                            "symptom you are experiencing."
                        )
                else:
                    return (
                        "Please answer <b>yes</b> or <b>no</b>."
                    )

            session["pending_symptom"] = None

        question = find_next_disease_question()

        if question:
            session["pending_symptom"] = question

            return (
                "Are you experiencing <b>" +
                clean_symp(question) +
                "</b>? Please answer yes or no."
            )

        # Current disease cannot be narrowed further.
        session["step"] = "PREDICT"
        return get_bot_response()

    # --------------------------------------------------------
    # IF NO MATCH: ASK FOR MORE SYMPTOMS
    # --------------------------------------------------------
    if session["step"] == "MORE_IF_NO_MATCH":

        extracted = extract_symptoms_from_message(s)

        if not extracted:
            extracted = local_symptom_fallback(s)

        if extracted:
            add_symptoms(extracted)
            session["step"] = "PD"
            return get_bot_response()

        return (
            "I still couldn't identify a supported symptom. "
            "Please describe another symptom, such as fever, "
            "headache, vomiting, cough, stomach pain, or fatigue."
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    if session["step"] == "PREDICT":

        prediction = predict_current_symptoms()

        if not prediction:
            session["step"] = "MORE_IF_NO_MATCH"
            return (
                "I couldn't generate a reliable prediction from the "
                "current symptoms. Please provide another symptom."
            )

        session["disease"] = prediction
        session["step"] = "DESCRIPTION"

        return build_final_prediction_response(prediction)

    # --------------------------------------------------------
    # DESCRIPTION / AI EXPLANATION
    # --------------------------------------------------------
    if session["step"] == "DESCRIPTION":

        save_assessment()
        session["step"] = "SEVERITY"

        disease_name = session["disease"]
        description = description_list.get(
            disease_name,
            "No detailed description is available."
        )

        return (
            "<b>About " + disease_name + ":</b><br>" +
            description +
            "<br><br>How many days have you had these symptoms?"
        )

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------
    if session["step"] == "SEVERITY":

        try:
            days = int(s)
        except ValueError:
            return (
                "Please enter the number of days, for example: 3."
            )

        if days < 1 or days > 3650:
            return "Please enter a valid number of days."

        session["days"] = days
        session["step"] = "FINAL"

        try:
            severe = calc_condition(
                session.get("all", []),
                days
            )
        except Exception:
            severe = 0

        if severe == 1:
            return (
                "<b>Please consult a qualified healthcare professional.</b>"
                "<br>Your symptom duration and severity indicate that "
                "professional medical evaluation is advisable."
                "<br><br>Tap <b>q</b> to exit."
            )

        precautions = precautionDictionary.get(
            session.get("disease"),
            []
        )

        message = (
            "Based on the available information, the model's prediction "
            "does not indicate a high severity score."
            "<br><br><b>Precautions:</b><br>"
        )

        if precautions:
            for i, precaution in enumerate(precautions, 1):
                message += f"{i}. {precaution}<br>"
        else:
            message += "No specific precautions are available in the dataset.<br>"

        message += (
            "<br>This is not a medical diagnosis. "
            "Please consult a healthcare professional if symptoms persist "
            "or worsen.<br><br>Tap <b>q</b> to end."
        )

        return message

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------
    if session["step"] == "FINAL":
        session["step"] = "BYE"
        return (
            "Your symptom assessment is complete. "
            "Would you like to start another assessment? "
            "(yes or no)"
        )

    # --------------------------------------------------------
    # ANOTHER ASSESSMENT
    # --------------------------------------------------------
    if session["step"] == "BYE":

        answer = normalize_yes_no(s)

        if answer is True:
            name = reset_diagnostic_keep_profile()
            return (
                "Hello again, " + str(name) +
                ". Please tell me your main symptom."
            )

        if answer is False:
            name = session.get("name", "User")
            session.clear()

            return (
                "Thank you, " + str(name) +
                ", for using Medibot. Take care! "
                "Please consult a qualified healthcare professional "
                "for diagnosis or treatment."
            )

        return "Please answer yes or no."

    # --------------------------------------------------------
    # SAFETY FALLBACK
    # --------------------------------------------------------
    return (
        "I couldn't continue the assessment from the current "
        "conversation state. Please refresh the page and start again."
    )


if __name__ == "__main__":
    app.run(debug=False)
