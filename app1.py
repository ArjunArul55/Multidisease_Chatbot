import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import csv
import json
import itertools
import os
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
    Use Gemini to convert natural language into symptoms
    that exist in the project's dataset.
    """
    if not message:
        return []

    symptom_names = ", ".join(all_symp_col)

    prompt = f"""
You are a symptom extraction system.

User message:
"{message}"

These are the ONLY symptoms supported by the machine-learning model:

{symptom_names}

Find the symptoms clearly mentioned by the user.

Rules:
1. Map natural language to the closest supported symptom.
2. Do not invent symptoms.
3. Do not diagnose a disease.
4. Only return symptoms from the supplied list.
5. Return JSON only, no preamble, no markdown formatting.

Required format:

{{
    "symptoms": ["symptom1", "symptom2"]
}}
"""

    result = ask_gemini(prompt)

    if not result:
        return []

    try:
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        data = json.loads(result)

        symptoms = data.get("symptoms", [])

        if not isinstance(symptoms, list):
            return []

        valid_symptoms = set(all_symp_col)

        return [
            symptom for symptom in symptoms
            if symptom in valid_symptoms
        ]

    except Exception as e:
        print("Symptom extraction error:", e)
        return []


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


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    s = request.args.get('msg')

    if "step" in session:
        if session["step"] == "Q_C":
            name = session["name"]
            age = session["age"]
            gender = session["gender"]
            session.clear()
            if s == "q":
                return "Thank you for using our web site Mr/Ms " + name
            else:
                session["step"] = "FS"
                session["name"] = name
                session["age"] = age
                session["gender"] = gender

    if s.upper() == "OK":
        return "What is your name ?"

    if 'name' not in session and 'step' not in session:

        # Handle first-message intent before treating the input as a name.
        intent = classify_intent(s)

        if intent == "greeting":
            return (
                "Hello! 👋 I'm your medical assistant. "
                "I can help analyze your symptoms using our "
                "machine-learning model. Please tell me what "
                "symptoms you're experiencing."
            )

        if intent == "irrelevant":
            return (
                "I'm designed to help with health-related symptoms "
                "and disease prediction. Please tell me about any "
                "symptoms you're experiencing."
            )

        if intent == "conversation":
            return ai_fallback_response(
                s,
                "The user has not started a diagnosis yet."
            )

        if intent == "medical":
            extracted = extract_symptoms_from_message(s)

            if extracted:
                session["all"] = extracted
                session["asked"] = []
                session["name"] = "User"
                session["step"] = "age"

                return (
                    "I detected these symptoms: "
                    + ", ".join(clean_symp(x) for x in extracted)
                    + ". How old are you?"
                )

            return (
                "I couldn't identify a supported symptom. "
                "Please describe what you are feeling."
            )

        session['name'] = s
        session['step'] = "age"
        return "How old are you?"

    if session["step"] == "age":

        try:
            age_text = ask_gemini(f'''
Extract the person's age from this message.

Message:
{s}

Return ONLY the integer age.
If no valid age is present, return 0.
''')

            age = int(age_text.strip()) if age_text else 0

        except Exception:
            age = 0

        if age < 1 or age > 120:
            return "Please enter your age, for example: 21."

        session["age"] = age
        session["step"] = "gender"

        return "Can you specify your gender?"

    if session["step"] == "gender":
        session["gender"] = s
        session["step"] = "Depart"

    if session['step'] == "Depart":
        session['step'] = "BFS"
        return "Well, Hello again Mr/Ms " + session[
            "name"] + ", now I will be asking some few questions about your symptoms to see what you should do. Tap S to start diagnostic!"

    if session['step'] == "BFS":
        session['step'] = "FS"
        return "Can you precise your main symptom Mr/Ms " + session["name"] + " ?"

    if session['step'] == "FS":
        # AI-powered guard: catch greetings / off-topic / quit / meta
        # questions before running them through the symptom pipeline.
        intent = classify_intent(s)
        if intent == "quit":
            session.clear()
            return "Alright, take care! Come back anytime you'd like a symptom check."
        if intent in ("greeting", "conversation", "irrelevant"):
            return ai_fallback_response(s, "The user is expected to describe a symptom at this stage.")

        raw1 = s
        sym1 = preprocess(s)
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr)
        temp = [sym1, sim1, psym1, raw1]
        session['FSY'] = temp
        session['step'] = "SS"
        if sim1 == 1:
            session['step'] = "RS1"
            resp = related_sym(psym1)
            if resp != 0:
                return resp
        else:
            return "You are probably facing another symptom, if so, can you specify it?"

    if session['step'] == "RS1":
        temp = session['FSY']
        psym1 = temp[2]
        psym1 = psym1[int(s)]
        temp[2] = psym1
        session['FSY'] = temp
        session['step'] = 'SS'
        return "You are probably facing another symptom, if so, can you specify it?"

    if session['step'] == "SS":
        raw2 = s
        sym2 = preprocess(s)
        sim2 = 0
        psym2 = []
        if len(sym2) != 0:
            intent = classify_intent(s)
            if intent == "quit":
                session.clear()
                return "Alright, take care! Come back anytime you'd like a symptom check."
            if intent in ("greeting", "conversation", "irrelevant"):
                return ai_fallback_response(s, "The user was asked if they have another symptom.")
            sim2, psym2 = syntactic_similarity(sym2, all_symp_pr)
        temp = [sym2, sim2, psym2, raw2]
        session['SSY'] = temp
        session['step'] = "semantic"
        if sim2 == 1:
            session['step'] = "RS2"
            resp = related_sym(psym2)
            if resp != 0:
                return resp

    if session['step'] == "RS2":
        temp = session['SSY']
        psym2 = temp[2]
        psym2 = psym2[int(s)]
        temp[2] = psym2
        session['SSY'] = temp
        session['step'] = "semantic"

    if session['step'] == "semantic":
        temp = session["FSY"]
        sim1 = temp[1]
        temp = session["SSY"]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
        else:
            session['step'] = 'PD'

    if session['step'] == "BFsim1=0":
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symp_pr)
            temp = session["FSY"]
            temp[1] = sim1
            temp[2] = psym1
            session['FSY'] = temp
            session['step'] = "sim1=0"
        else:
            session['step'] = "BFsim2=0"

    if session['step'] == "sim1=0":
        temp = session["FSY"]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if s == "yes":
                    psym1 = sugg[0]
                    sim1 = 1
                    temp = session["FSY"]
                    temp[1] = sim1
                    temp[2] = psym1
                    session["FSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested" not in session:
                sym1 = session["FSY"][0]
                session["suggested"] = suggest_syn(sym1)
                sugg = session["suggested"]
            if len(sugg) > 0:
                session["suggested"] = sugg
                msg = "are you experiencing any  " + sugg[0] + "?"
                return msg
        if "suggested" in session:
            del session["suggested"]
        session['step'] = "BFsim2=0"

    if session['step'] == "BFsim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symp_pr)
            temp = session["SSY"]
            temp[1] = sim2
            temp[2] = psym2
            session['SSY'] = temp
            session['step'] = "sim2=0"
        else:
            session['step'] = "TEST"

    if session['step'] == "sim2=0":
        temp = session["SSY"]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if s == "yes":
                    psym2 = sugg[0]
                    sim2 = 1
                    temp = session["SSY"]
                    temp[1] = sim2
                    temp[2] = psym2
                    session["SSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                sym2 = session["SSY"][0]
                session["suggested_2"] = suggest_syn(sym2)
                sugg = session["suggested_2"]
            if len(sugg) > 0:
                msg = "Are you experiencing " + sugg[0] + "?"
                session["suggested_2"] = sugg
                return msg
        if "suggested_2" in session:
            del session["suggested_2"]
        session['step'] = "TEST"

    if session['step'] == "TEST":
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        raw1 = temp[3] if len(temp) > 3 else ""
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        raw2 = temp[3] if len(temp) > 3 else ""

        if sim1 == 0 and sim2 == 0:
            # Deterministic pipeline (syntactic + semantic similarity)
            # found nothing. Fall back to Gemini to try to map the raw
            # free-text messages onto known dataset symptoms.
            raw_combined = (str(raw1) + " " + str(raw2)).strip()
            ai_symptoms = extract_symptoms_from_message(raw_combined)
            if ai_symptoms:
                session["all"] = ai_symptoms
                session["asked"] = []
                session['step'] = 'PD'
            else:
                result = None
                session["offtopic_msg"] = ai_fallback_response(raw_combined) if raw_combined else None
                session['step'] = "END"
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["FSY"]
                temp[2] = psym2
                session["FSY"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["SSY"]
                temp[2] = psym1
                session["SSY"] = temp
            session['step'] = 'PD'

    if session['step'] == 'PD':
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if "all" not in session:
            session["asked"] = []
            session["all"] = [col_dict[psym1], col_dict[psym2]]
        session["diseases"] = possible_diseases(session["all"])
        diseases = session["diseases"]
        if diseases:
            dis = diseases[0]
            session["dis"] = dis
        session['step'] = "for_dis"

    if session['step'] == "DIS":
        symts = session.get("symv", [])
        if "symv" in session:
            if len(s) > 0 and len(symts) > 0:
                all_sym = session["all"]
                if s == "yes":
                    all_sym.append(symts[0])
                    session["all"] = all_sym
                if symts:
                    del symts[0]
                session["symv"] = symts
        if "symv" not in session:
            session["symv"] = symVONdisease(df_tr, session["dis"])
            symts = session["symv"]
        if len(symts) > 0:
            if symts[0] not in session["all"] and symts[0] not in session["asked"]:
                asked = session["asked"]
                asked.append(symts[0])
                session["asked"] = asked
                msg = "Are you experiencing " + clean_symp(symts[0]) + "?"
                return msg
            else:
                del symts[0]
                session["symv"] = symts
                s = ""
                return get_bot_response()
        else:
            PD = possible_diseases(session["all"])
            diseases = session["diseases"]
            if diseases and diseases[0] in PD:
                session["testpred"] = diseases[0]
                PD.remove(diseases[0])
            session["diseases"] = PD
            session['step'] = "for_dis"

    if session['step'] == "for_dis":
        diseases = session["diseases"]
        if len(diseases) <= 0:
            session['step'] = 'PREDICT'
        else:
            session["dis"] = diseases[0]
            session['step'] = "DIS"
            session["symv"] = symVONdisease(df_tr, session["dis"])
            return get_bot_response()

    if session['step'] == "PREDICT":
        result = knn_clf.predict(OHV(session["all"], all_symp_col))
        session['step'] = "END"

    if session['step'] == "END":
        if result is not None:
            if "testpred" in session and result[0] != session["testpred"]:
                session['step'] = "Q_C"
                return "as you provide me with few symptoms, I am sorry to announce that I cannot predict your " \
                       "disease for the moment!!! <br> Can you specify more about what you are feeling or Tap q to " \
                       "stop the conversation "
            session['step'] = "Description"
            session["disease"] = result[0]
            return (
                "The machine-learning model predicts <b>" + result[0] +
                "</b> based on the symptoms you provided. "
                "This is not a confirmed medical diagnosis. "
                "Tap D to continue with the explanation and precautions."
            )
        else:
            offtopic_msg = session.pop("offtopic_msg", None)
            session['step'] = "Q_C"
            if offtopic_msg:
                return offtopic_msg + "<br>Can you specify more about what you are feeling, or tap q to stop the conversation."
            return ("as you provide me with few symptoms, I am sorry to announce that I cannot predict your "
                    "disease for the moment!!! <br> Can you specify more about what you are feeling or Tap q to "
                    "stop the conversation ")

    if session['step'] == "Description":
        y = {"Name": session["name"], "Age": session["age"], "Gender": session["gender"],
             "Disease": session["disease"], "Sympts": session["all"]}
        write_json(y)
        session['step'] = "Severity"

        # Prefer a Gemini-generated, friendlier explanation of the
        # prediction; fall back to the raw dataset description if the
        # API is unavailable.
        ai_explanation = generate_ai_response(session["disease"], session["all"])
        if ai_explanation:
            return ai_explanation + " <br> How many days have you had symptoms?"

        if session["disease"] in description_list.keys():
            return description_list[session["disease"]] + " \n <br>  How many days have you had symptoms?"
        else:
            disease_slug = session["disease"]
            if " " in disease_slug:
                disease_slug = disease_slug.replace(" ", "_")
            return "please visit <a href='" + "https://en.wikipedia.org/wiki/" + disease_slug + "'>  here  </a>"

    if session['step'] == "Severity":
        session['step'] = 'FINAL'
        if calc_condition(session["all"], int(s)) == 1:
            return "you should take the consultation from doctor <br> Tap q to exit"
        else:
            msg = 'Nothing to worry about, but you should take the following precautions :<br> '
            i = 1
            for e in precautionDictionary.get(session["disease"], []):
                msg += '\n ' + str(i) + ' - ' + e + '<br>'
                i += 1
            msg += ' Tap q to end'
            return msg

    if session['step'] == "FINAL":
        session['step'] = "BYE"
        return (
            "Your symptom assessment is complete. "
            "Would you like to start another assessment (yes or no)?"
        )

    if session['step'] == "BYE":
        name = session["name"]
        age = session["age"]
        gender = session["gender"]
        session.clear()
        if s.lower() == "yes":
            session["gender"] = gender
            session["name"] = name
            session["age"] = age
            session['step'] = "FS"
            return "HELLO again Mr/Ms " + session["name"] + " Please tell me your main symptom. "
        else:
            return (
                "Thank you, " + name +
                ", for using the medical assistant. "
                "Please consult a qualified healthcare professional "
                "for diagnosis or treatment."
            )

    # Safety net: if we somehow fall through every branch above without
    # returning, don't crash — hand off to Gemini instead of a 500 error.
    return ai_fallback_response(s or "", "The chatbot's internal state machine did not match a known step.")


if __name__ == "__main__":
    app.run(debug=False)
