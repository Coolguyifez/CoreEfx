from flask import Flask, request, redirect, render_template_string, render_template, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import spacy
# === NEW Machine Learning imports ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#=== Voice import===
from gtts import gTTS
import uuid, os
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user, UserMixin
)
from flask_bcrypt import Bcrypt




# === gTTS helper ===
def generate_audio(text):
    # Create a unique filename for each response
    filename = f"static/audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Initialize the Flask application
app = Flask(__name__)

# Load a larger spaCy model if you want to use semantic similarity later,
# otherwise, 'en_core_web_sm' is fine for POS and lemmatization.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Configure SQLite database.
# The database file 'symptoms.db' will be created in the same directory as this script.
db_path = "postgresql://symptom_db_user:HtANNPQJRywNPbHqJxYnxcyqi8GzRWre@dpg-d43r7h2dbo4c73ausln0-a.oregon-postgres.render.com/symptom_db"
app.secret_key = "super_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = db_path
# This setting is to suppress a warning from SQLAlchemy; it's recommended to set it to False.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Initialize the SQLAlchemy database object, linking it to the Flask app.
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Define Database Models
# These classes represent tables in your SQLite database.
# === Flask-Login setup ===
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === User model ===
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class SymptomReport(db.Model):
    """
    Represents a record of a user's symptom submission, including their input,
    detected location, the advice given, and a timestamp.
    """
    id = db.Column(db.Integer, primary_key=True)  # Unique identifier for each report
    input_text = db.Column(db.String(500))  # The raw text input from the user (symptoms)
    location = db.Column(db.String(100))  # User's approximate location (latitude,longitude string)
    result = db.Column(db.String(500))  # The health advice/diagnosis provided by the system (increased length)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Time when the report was created
    # 🔑 Link report to user
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('reports', lazy=True))

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    action = db.Column(db.String(20))  # "login" or "logout"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    """
    Represents a record for user feedback.
    """
    id = db.Column(db.Integer, primary_key=True)  # Unique identifier for each feedback entry
    message = db.Column(db.String(1000))  # The feedback message content
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class Hospital(db.Model):
    """
    Represents a hospital entry, storing its name, city, and geographical coordinates,
    and a URL for more information.
    """
    id = db.Column(db.Integer, primary_key=True)  # Unique identifier for each hospital
    name = db.Column(db.String(200), nullable=False)  # Name of the hospital (cannot be null)
    city = db.Column(db.String(100), nullable=False)  # City where the hospital is located (cannot be null)
    lat = db.Column(db.Float, nullable=True)  # Latitude of the hospital (can be null if not available)
    lon = db.Column(db.Float, nullable=True)  # Longitude of the hospital (can be null if not available)
    url = db.Column(db.String(500), nullable=True)  # URL to the hospital's website or info page


# Create Database Tables and Seed Initial Data
# This block runs within the Flask application context to interact with the database.
with app.app_context():
    db.create_all()  # Creates all tables defined by the SQLAlchemy models if they don't already exist.

    # Check if the Hospital table is empty. If so, populate it with sample hospital data.
    if Hospital.query.count() == 0:
        sample_hospitals = [
            # Hospitals in major Nigerian cities
            Hospital(name="Lagos State University Teaching Hospital (LASUTH)", city="lagos", lat=6.59047449787585,   lon=3.3422608588498037,
                     url="https://lasuth.org.ng/"),
            Hospital(name="University of Port Harcourt Teaching Hospital", city="choba port harcourt", lat=4.903006121694171,  lon=6.928573993848487,
                     url="https://www.google.com/search?q=upth+port+harcourt&rlz"),
            Hospital(name="University of Abuja Specialist Hospital", city="abuja", lat=8.965031007601917,  lon=7.064360769741204,
                     url="https://www.google.com/search?q=university+of+abuja+specialist+hospital+gwagwalada&sca"),

            # Hospitals near Delta State (specifically added for your location context)
            Hospital(name="Central Hospital Warri", city="warri", lat=5.517178270540105, lon=5.734592093139206,
                     url="https://www.google.com/search?q=central+hospital+warri&sca"),
            Hospital(name="Delta State University Teaching Hospital", city="Oghara", lat=5.9602250750294345 , lon= 5.702942682383533,
                     url="https://www.google.com/search?q=delta+state+university+teaching+hospital&sca"),  # Located in Oghara
            Hospital(name="Federal Medical Centre Asaba", city="asaba", lat=6.2121256496757615, lon=6.7122751952742465,
                     url="https://www.google.com/search?q=Federal+Medical+Centre+Asaba&sca"),
            Hospital(name="University of Benin Teaching Hospital (UBTH)", city="benin city", lat=6.3903335466504725, lon=5.611826787114936,
                     url="https://www.google.com/search?q=ubth&sca"),
            Hospital(name="Federal Medical Centre Yenagoa", city="yenagoa", lat=4.937291400004392, lon=6.266638050617462,
                     url="https://www.google.com/search?q=federal+medical+centre+yenagoa&sca"),
            Hospital(name="Divine Grace Hospital Warri", city="warri", lat=5.568987307199267,  lon=5.767263095268969, url="https://www.bing.com/search?q=divine_grace_hospital%2ceffurun.delta_state&qs"),
            Hospital(name="Lily Hospitals Ltd. Warri", city="warri", lat=5.526185526750632,  lon=5.765111210609841,
                     url="https://lilyhospitals.com/"),
            Hospital(name="Asaba Specialist Hospital", city="asaba", lat=6.234913709017079,  lon=6.685982039450925, url="https://asabaspecialisthospital.org/")
        ]
        # Add all sample hospital objects to the current database session.
        db.session.bulk_save_objects(sample_hospitals)
        db.session.commit()  # Commit the changes, saving the hospitals to the database.
# === NEW: Machine Learning Setup ===
ml_vectorizer = None
ml_model = None

def train_ml_model():
    """
    Train a simple ML classifier based on symptom_data dictionary
    """
    global ml_vectorizer, ml_model
    train_texts, train_labels = [], []
    for condition, details in symptom_data.items():
        combined_text = " ".join(details["symptoms"])
        train_texts.append(combined_text)
        train_labels.append(condition)
    ml_vectorizer = TfidfVectorizer()
    X = ml_vectorizer.fit_transform(train_texts)
    ml_model = LogisticRegression(max_iter=1000)
    ml_model.fit(X, train_labels)
    print("✅ ML model trained on symptom_data")


def ml_predict_condition(text, top_n=5, threshold=0.0):
    """Return top N possible conditions with probability >= threshold"""
    if not ml_model:
        return []

    X_user = ml_vectorizer.transform([text])
    proba = ml_model.predict_proba(X_user)[0]

    # Pair condition + probability
    cond_probs = list(zip(ml_model.classes_, proba))
    # Sort descending by prob
    cond_probs.sort(key=lambda x: x[1], reverse=True)

    # Filter above threshold
    filtered = [(cond, p) for cond, p in cond_probs if p >= threshold]

    # Return top N
    return filtered[:top_n]

# Predefined Symptom Data and Advice
# This dictionary holds various health conditions, their associated keywords/symptoms,
# and the corresponding advice to be given to the user.
symptom_data = {
    "malaria": {
        "symptoms": [
            "fever (body hot / body dey hot)",
            "chills (body dey shake)",
            "sweating (too much sweat)",
            "headache (head dey pain)",
            "fatigue (body weak)",
            "nausea (body dey turn)",
            "vomiting (dey vomit)",
            "muscle aches (body pain)"
        ],
        "advice": "\nYou may have malaria. Please visit the health center quickly for a blood test and proper malaria treatment."
    },

    "typhoid fever": {
        "symptoms": [
            "fever wey last many days",
            "stomach pain (belly dey pain)",
            "diarrhea (stool dey run)",
            "constipation (no fit stool)",
            "weakness (body weak well well)",
            "loss of appetite (no wan chop)"
        ],
        "advice": "\nThis may be typhoid fever. Drink only safe water and visit the nearest health center for a test and antibiotics."
    },

    "common cold": {
        "symptoms": [
            "cough (dey cough)",
            "sore throat (throat dey pain)",
            "runny nose (catarrh dey come out)",
            "nasal congestion (nose block)",
            "mild fever (small body hot)"
        ],
        "advice": "\nThis looks like a common cold. Rest, drink plenty of fluids, and see a health worker if it worsens."
    },

    "influenza (flu)": {
        "symptoms": [
            "high fever (body dey hot well well)",
            "body aches (body pain)",
            "headache (head dey pain)",
            "cough (dey cough)",
            "fatigue (no strength)"
        ],
        "advice": "\nThis could be influenza (flu). Rest, drink fluids, and visit a health worker if breathing is difficult."
    },

    "diarrheal disease": {
        "symptoms": [
            "diarrhea (stool dey rush)",
            "vomiting (dey vomit)",
            "dehydration (mouth dry)",
            "stomach cramps (belly twist)"
        ],
        "advice": "\nThis may be a diarrheal infection. Start oral rehydration solution immediately and see a health worker if it continues."
    },

    "urinary tract infection (UTI)": {
        "symptoms": [
            "pain when urinating (when you piss e dey pain)",
            "frequent urge to urinate (dey always wan piss)",
            "lower stomach pain (lower belly dey pain)"
        ],
        "advice": "\nThis looks like a urinary tract infection. Drink clean water and visit a health center for proper antibiotics."
    },

    "skin infection (rash/measles/chickenpox)": {
        "symptoms": [
            "rash (skin get small small spots)",
            "itchy skin (skin dey scratch)",
            "blisters (skin get water blister)",
            "red spots (red mark for body)",
            "fever (body hot)"
        ],
        "advice": "\nThis may be a skin infection or measles. Avoid scratching, keep clean, and visit a health center for assessment."
    },

    "dehydration/heat exhaustion": {
        "symptoms": [
            "excessive thirst (mouth dey dry)",
            "dry mouth (tongue dry)",
            "dizziness (head dey turn)",
            "weakness (no strength)",
            "no urine (no dey piss)"
        ],
        "advice": "\nYou may be dehydrated. Drink clean water or oral rehydration solution. Seek care if severe."
    },

    "headache": {
        "symptoms": [
            "headache (head dey pain)",
            "pressure in head",
            "migraine"
        ],
        "advice": "\nThis may be due to stress, minor infection, or malaria. Rest, stay hydrated, and seek care if it persists."
    },

    "body pain": {
        "symptoms": [
            "muscle pain",
            "joint pain",
            "back pain"
        ],
        "advice": "\nThis may be due to fatigue, infection, or malaria. Rest and consult a doctor if it persists."
    },

    "early pregnancy": {
        "symptoms": [
            "missed period (menstruation no come)",
            "nausea (body dey turn)",
            "vomiting (dey vomit for morning)",
            "breast tenderness (breast dey pain)",
            "fatigue (no strength)"
        ],
        "advice": "\nThese could be early pregnancy signs. Visit a health center for confirmation and antenatal care."
    },

    "eye infection (Apollo/conjunctivitis)": {
        "symptoms": [
            "red eye (eye red)",
            "itchy eye (eye dey scratch)",
            "watery eye (eye dey bring water)",
            "eye pain (eye dey pain)",
            "sticky discharge (eye gum when you wake)"
        ],
        "advice": "\nThis may be conjunctivitis (Apollo). Avoid touching your eyes, wash your hands, and visit a health center for proper eye drops."
    },

    "blurred vision/eye problem": {
        "symptoms": [
            "blurred vision (no see clear)",
            "eye pain (eye dey pain)",
            "light sensitivity (eye no like light)",
            "vision loss (eye dey blind small small)"
        ],
        "advice": "\nYour eyes need urgent check-up. Do not wait until it worsens. Visit a health center or eye clinic immediately."
    },
    "pneumonia": {
        "symptoms": [
            "fast breathing (dey breathe fast)",
            "difficulty breathing (no fit breathe well)",
            "cough with mucus (cough dey bring phlegm)",
            "high fever (body hot well well)",
            "chest pain (chest dey pain)"
        ],
        "advice": "\nThis could be pneumonia, especially in children or elderly. Go to a health center immediately."
    },

    "hiv/aids": {
        "symptoms": [
            "long lasting fever (body hot for weeks)",
            "weight loss (body dey reduce)",
            "night sweats (body dey sweat for night)",
            "persistent diarrhea (stool no dey stop)",
            "weakness (body weak)"
        ],
        "advice": "\nThis may be HIV or another serious infection. Please visit a health center for proper testing and early treatment."
    },

    "tuberculosis (tb)": {
        "symptoms": [
            "cough wey last more than 2 weeks",
            "coughing blood (cough dey bring blood)",
            "weight loss (body dey reduce)",
            "night sweats (body dey sweat for night)",
            "chest pain (chest dey pain)"
        ],
        "advice": "\nThis could be tuberculosis . It can spread to others. Go to an health center immediately for free testing and treatment."
    },

    "diabetes": {
        "symptoms": [
            "frequent urination (dey piss too much)",
            "excessive thirst (always dey thirsty)",
            "always hungry (too much hunger)",
            "weight loss (body dey slim down)",
            "wounds no dey heal quick"
        ],
        "advice": "\nThis may be diabetes. Avoid too much sugar and visit a health center for a blood sugar test."
    },

    "hypertension (high blood pressure)": {
        "symptoms": [
            "headache (head dey pain often)",
            "dizziness (head dey turn)",
            "blurred vision (eye no see clear)",
            "chest pain (chest dey pain)",
            "sometimes no symptoms at all"
        ],
        "advice": "\nThis may be high blood pressure. Check it at a health center. Reduce salt intake and stress."
    },

    "heart disease": {
        "symptoms": [
            "chest pain (chest dey pain or tight)",
            "shortness of breath (no fit breathe well)",
            "swelling in legs (leg dey swell)",
            "tired easily (body weak quick quick)",
            "fast or irregular heartbeat (heart dey beat fast)"
        ],
        "advice": "\nThis may be a heart problem. Please go to a health center immediately for a check-up."
    },

    "malnutrition": {
        "symptoms": [
            "weight loss (body too slim)",
            "swollen feet (leg dey swell from kwashiorkor)",
            "thin arms (hand dey thin)",
            "child no grow well",
            "weakness (no strength)"
        ],
        "advice": "\nThis person may be malnourished. Provide balanced meals and visit a health center for nutrition support."
    },
}




def check_symptoms(user_input, min_score_threshold=1, top_n=5):
    """
    Analyzes user symptom input with spaCy lemmas & scores each condition.
    Returns top N conditions sorted by highest match score.
    """
    doc_user_input = nlp(user_input.lower())
    user_tokens = {token.lemma_ for token in doc_user_input if token.pos_ in ['NOUN', 'ADJ', 'VERB']}

    condition_scores = {}

    for condition, details in symptom_data.items():
        current_condition_score = 0
        for defined_symptom_phrase in details["symptoms"]:
            doc_defined_symptom = nlp(defined_symptom_phrase.lower())
            defined_lemmas = {token.lemma_ for token in doc_defined_symptom if token.pos_ in ['NOUN', 'ADJ', 'VERB']}

            # Count shared lemmas
            common_lemmas = user_tokens.intersection(defined_lemmas)

            if common_lemmas:
                # Each matching symptom phrase adds +1 score
                current_condition_score += 1

        if current_condition_score >= min_score_threshold:
            condition_scores[condition] = current_condition_score

    # Sort by highest score
    sorted_conditions = sorted(condition_scores.items(), key=lambda item: item[1], reverse=True)

    # Return top N conditions with scores
    return sorted_conditions[:top_n]

def normalize_condition_name(name: str) -> str:
    """Standardize condition names to match symptom_data keys."""
    return name.strip().lower()

medical_keywords = [
    "fever", "body hot", "body dey hot", "chills", "body dey shake", "sweating", "too much sweat",
    "headache", "head dey pain", "fatigue", "body weak", "no strength", "nausea", "body dey turn",
    "vomiting", "dey vomit", "muscle aches", "body pain", "stomach pain", "stomach cramp", "belly twist"  "belly dey pain",
    "diarrhea", "stool dey run", "stool dey rush", "constipation", "no fit stool", "weakness",
    "body weak well well", "loss of appetite", "no wan chop", "cough", "dey cough", "sore throat",
    "throat dey pain", "runny nose", "catarrh dey come out", "nasal congestion", "nose block",
    "mild fever", "small body hot", "high fever", "body dey hot well well", "body aches",
    "pressure in head", "migraine", "joint pain", "back pain", "waist dey pain", "missed period",
    "menstruation no come", "breast tenderness", "breast dey pain", "red eye", "eye red", "itchy eye",
    "eye dey scratch", "watery eye", "eye dey bring water", "eye pain", "eye dey pain", "sticky discharge",
    "eye gum when you wake", "blurred vision", "no see clear", "light sensitivity", "eye no like light",
    "vision loss", "eye dey blind small small", "fast breathing", "dey breathe fast", "difficulty breathing",
    "no fit breathe well", "cough with mucus", "cough dey bring phlegm", "chest pain", "chest dey pain",
    "chest dey pain or tight", "long lasting fever", "body hot for weeks", "weight loss", "body dey reduce",
    "night sweats", "body dey sweat for night", "persistent diarrhea", "stool no dey stop", "frequent urination",
    "dey piss too much", "excessive thirst", "always dey thirsty", "always hungry", "too much hunger",
    "wounds no dey heal quick", "sometimes no symptoms at all", "shortness of breath", "swelling in legs",
    "leg dey swell", "tired easily", "body weak quick quick", "fast heartbeat", "irregular heartbeat",
    "heart dey beat fast", "swollen feet", "leg dey swell from kwashiorkor", "thin arms", "hand dey thin",
    "child no grow well"
]

def has_medical_relevance(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in medical_keywords)


train_ml_model()

# --- Example Usage (assuming you have your symptom_data loaded elsewhere) ---
# user_input_example = "I have fever, chills and loss of appetite"
# results = check_symptoms(user_input_example)
# print(results)
#
# # Expected output with the provided symptom_data:
# # ['chickenpox (varicella)', 'infectious mononucleosis (mono/kissing disease)', 'malaria']
# # (Order might vary as sets don't maintain order)

def find_nearby_hospitals(user_lat, user_lon, radius_km=200):
    """
    Finds hospitals within a specified geographical radius from a given point.
    Returns a list of dictionaries with hospital name, city, lat, lon, and url.

    Args:
        user_lat (float): Latitude of the user's current location.
        user_lon (float): Longitude of the user's current location.
        radius_km (int, optional): The radius in kilometers to search for hospitals. Defaults to 100.

    Returns:
        list: A list of dictionaries, each representing a hospital.
    """
    nearby = []
    # Retrieve all hospital records from the database.
    for hospital in Hospital.query.all():
        # Ensure the hospital has valid latitude and longitude coordinates.
        if hospital.lat is not None and hospital.lon is not None:
            # Calculate the Euclidean distance between the user and the hospital's coordinates.
            # Approximation: 1 degree latitude ~ 111 km. Rough for longitude, especially away from equator.
            distance_in_degrees = ((user_lat - hospital.lat) ** 2 + (user_lon - hospital.lon) ** 2) ** 0.5
            distance_in_km = distance_in_degrees * 111
            # If the hospital is within the specified radius, add it to the nearby list.
            if distance_in_km <= radius_km:
                # Return as a dictionary suitable for JSON serialization and JavaScript use
                nearby.append({
                    'name': hospital.name,
                    'city': hospital.city,
                    'lat': hospital.lat,
                    'lon': hospital.lon,
                    'url': hospital.url
                })
    return nearby

# === NEW Welcome Template ===
welcome_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Welcome to CoreEfx AI</title>
    <link rel="icon" type="image/png" href="{{ url_for('static', filename='images/brain.png') }}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" />

    <style>
        :root {
            --transition-speed: 0.3s;
            --light-bg: #ffffff;
            --light-text: #008000;
            --dark-bg: #2c2f33;
            --dark-text: #38b45a;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: 'Roboto', sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: var(--light-bg);
            color: var(--light-text);
            text-align: center;
            transition: background var(--transition-speed), color var(--transition-speed);
        }

        h1 {
            font-size: clamp(2rem, 8vw, 5rem);
            font-weight: 720;
            line-height: 1.2;
            margin-bottom: -10px;
            margin-top: -5px;
            opacity: 0; /* Start hidden for animation */
            transform: translateY(20px); /* Start slightly below for animation */
            animation: fadeInScale 3s ease-out forwards; /* Apply animation */
        }

        /* Animation Keyframes */
        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        h3 {
            font-size: 30px;
            font-weight: 1;
            opacity: 0; /* Start hidden for animation */
            transform: translateY(20px); /* Start slightly below for animation */
            animation: fadeInScale 3s ease-out forwards; /* Apply animation */
       }
       @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.75);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(0.95);
            }
        }

        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }


       

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            body {
                background: var(--dark-bg);
                color: var(--dark-text);
            }
           
        }

        /* Extra small screen tweaks */
        @media (max-width: 780px) {
            h1 {
                font-size: clamp(2.2rem, 12vw, 2.2rem);
                 
            }
            
            h3 {
                font-size: 20px;
            }
           
        }
         @media (max-width: 480px) {
           h1 {
                font-size: clamp(1.5rem, 10vw, 1.5rem);
                
                
            }
            h3{
                font-size: 14px;
               
            }
            
        }
    </style>
</head>
<body>
    <h1><i class="fa-solid fa-brain"></i>CoreEfx AI</h1>
     <h3><i>Check Symptoms, Stay Healthy<i></h3>
    
</body>
</html>
"""

# === Login, Signup Templates ===
login_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CoreEfx AI Login</title>
  <link rel="icon" type="image/png" href="{{ url_for('static', filename='images/brain.png') }}">

  <!-- Leaflet & FontAwesome -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
  <style>
    html {
      box-sizing: border-box;
    }
    *, *:before, *:after {
      box-sizing: inherit;
    }

    body {
      font-family: Arial, sans-serif;
      background: #f5f5f5;
      color: #333;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }

    form {
      background: white;
      padding: 30px;
      width: 100%;
      max-width: 420px;
      border-radius: 20px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    h1 {
      color: #008000;
      margin-bottom: 20px;
      text-align: center;
    }

    input {
      width: 100%;
      padding: 15px;
      margin: 12px 0;
      border: 1px solid #ccc;
      border-radius: 20px;
      font-size: 16px;
    }

    button {
      width: 100%;
      padding: 15px;
      background: #008000;
      color: white;
      border: none;
      border-radius: 20px;
      font-size: 17px;
      font-weight: bold;
      cursor: pointer;
    }

    button:hover {
      background: #28a428;
    }

    p {
      margin-top: 15px;
      font-size: 15px;
      text-align: center;
    }

    a {
      color: #008000;
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    /* Eye icon positioning */
    .input-group {
      position: relative;
    }
    .input-group .icon {
        position: absolute;
        top: 50%;
        left: 15px;
        transform: translateY(-50%);
        color: #008000;
        font-size: 16px;
    }

    .input-group input {
        padding-left: 45px;
      padding-right: 100px; /* Make space for eye icon */
    }

    .toggle-password {
      position: absolute;
      top: 50%;
      left: auto;
      right: 15px;
      transform: translateY(-50%);
      background: none;
      border: none;
      cursor: pointer;
      font-size: 18px;
      color: #008000;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      margin: 0;
    }

    .toggle-password:hover {
      color: #28a428;
      background: none;
      boarder:none;
    }

    /* ✅ Dark Mode */
    @media (prefers-color-scheme: dark) {
      body { background: #2c2f33; color: #eee; }
      form { background: #36393f; box-shadow: none; }
      input { background: #2c2f33; color: #eee; border: 1px solid #444; }
      input::placeholder { color: #aaa; }
      button { background: #38b45a; color: white; }
      a { color: #38b45a; }
      h1 { color:#38b45a;}
      a:hover { color:#2e8b43;}
      button:hover { background:#2e8b43;}
      .toggle-password { color:#38b45a;}
      .toggle-password:hover{ color:#2e8b43;}
      .input-group .icon { color:#38b45a;}
    }

    /* ✅ Mobile view adjustments */
    @media (max-width: 768px) {
      body {
        padding: 0;
      }

      form {
       
        width: 100%;
        margin: 0 auto;
        border-radius: 10px;
        /*height: 150vh;*/
        box-shadow: none;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 40px 10px;
      }

      input, button {
        font-size: 15px;
        padding: 12px;
      }

      h1 {
        font-size: 36px;
        margin-bottom: 40px;
      }
      .toggle-password{
        right: 10px;
        left: auto;
      }
      
    }

    @media (max-width: 480px) {
      form {
        padding: 300px 18px;
        margin: 0 auto;
        position: fixed;
      }

      input, button {
        font-size: 14px;
        padding: 12px;
      }

      h1{
        font-size: 34px;
      }
    }
  </style>
</head>
<body>
  
    <form method="POST">
     <h1><i class="fa-solid fa-brain fa-icon-large"></i>CoreEfx AI</h1>
      <div class="input-group">
        <i class="fa fa-user icon"></i>
        <input type="username" name="username" placeholder="Username" required />
      </div>
      <div class="input-group">
        <i class="fa fa-lock icon"></i>
        <input type="password" id="password" name="password" placeholder="Password" required />
        <div type="button" class="toggle-password" onclick="togglePassword()">
            <i class="fa-solid fa-eye" id="eye-icon"></i>
        </div>
      </div>
      <p style="margin-top: 10px;">
        <a href="/forgot-password">Forgot Password?</a>
      </p>
      <button type="submit">Login</button>
    <p>
      Don't have an account? <a href="/signup">Sign Up</a>
    </p>
    
    </form>
 <script>
    function togglePassword() {
      const password = document.getElementById("password");
      const icon = document.getElementById("eye-icon");

      if (password.type === "password") {
        password.type = "text";
        icon.classList.remove("fa-eye");
        icon.classList.add("fa-eye-slash");
      } else {
        password.type = "password";
        icon.classList.remove("fa-eye-slash");
        icon.classList.add("fa-eye");
      }
    }
 </script>
</body>
</html>
"""
signup_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CoreEfx AI Sign Up</title>
  <link rel="icon" type="image/png" href="{{ url_for('static', filename='images/brain.png') }}">

  <!-- Leaflet & FontAwesome -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">  

  <style>
    html {
      box-sizing: border-box;
    }
    *, *:before, *:after {
      box-sizing: inherit;
    }

    body {
      font-family: Arial, sans-serif;
      background: #f5f5f5;
      color: #333;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }

    form {
      background: white;
      padding: 30px;
      width: 100%;
      max-width: 420px;
      border-radius: 20px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    h1 {
      color: #008000;
      margin-bottom: 20px;
      text-align: center;
    }

    input {
      width: 100%;
      padding: 15px;
      margin: 12px 0;
      border: 1px solid #ccc;
      border-radius: 20px;
      font-size: 16px;
    }

    button {
      width: 100%;
      padding: 15px;
      background: #008000;
      color: white;
      border: none;
      border-radius: 20px;
      font-size: 17px;
      font-weight: bold;
      cursor: pointer;
    }

    button:hover {
      background: #28a428;
    }

    p {
      margin-top: 15px;
      font-size: 15px;
      text-align: center;
    }

    a {
      color: #008000;
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    /* Eye icon positioning */
    .input-group {
      position: relative;
    }
    
    .input-group .icon {
        position: absolute;
        top: 50%;
        left: 15px;
        transform: translateY(-50%);
        color: #008000;
        font-size: 16px;
    }

    .input-group input {
      padding-left: 45px;
      padding-right: 100px; /* Make space for eye icon */
    }

    .toggle-password {
      position: absolute;
      top: 50%;
      left: auto;
      right: 15px;
      transform: translateY(-50%);
      background: none;
      border: none;
      cursor: pointer;
      font-size: 18px;
      color: #008000;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      margin: 0;
    }

    .toggle-password:hover {
      color: #28a428;
      background: none;
      border:none;
    }

    /* ✅ Dark Mode */
    @media (prefers-color-scheme: dark) {
      body { background: #2c2f33; color: #eee; }
      form { background: #36393f; box-shadow: none; }
      input { background: #2c2f33; color: #eee; border: 1px solid #444; }
      input::placeholder { color: #aaa; }
      button { background: #38b45a; color: white; }
      a { color: #38b45a; }
      h1 { color:#38b45a;}
      a:hover { color:#2e8b43;}
      button:hover { background:#2e8b43;}
      .toggle-password { color:#38b45a;}
      .toggle-password:hover{ color:#2e8b43;}
      .input-group .icon { color:#38b45a;}
    }

    /* ✅ Mobile view adjustments */
    @media (max-width: 768px) {
      body {
        padding: 0;
      }

      form {
       
        width: 100%;
        margin: 0 auto;
        border-radius: 10px;
        /*height: 150vh;*/
        box-shadow: none;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 40px 10px;
      }

      input, button {
        font-size: 15px;
        padding: 12px;
      }

      h1 {
        font-size: 36px;
        margin-bottom: 30px;
      }
      
      .toggle-password{
        right: 10px;
        left: auto;
      }
    }

    @media (max-width: 480px) {
      form {
        padding: 300px 18px;
        margin: 0 auto;
        position: fixed;
        
        
      }

      input, button {
        font-size: 14px;
        padding: 12px;
      }

      h1 {
        font-size: 34px;
      }
    }
  </style>
</head>
<body>
  <form method="POST">
     <h1><i class="fa-solid fa-brain fa-icon-large"></i>CoreEfx AI</h1>
    <div class="input-group">
        <i class="fa fa-user icon"></i>
        <input type="text" name="name" placeholder="Fullname" required>
    </div>
    <div class="input-group">
        <i class="fa fa-user icon"></i>
        <input type="text" name="username" placeholder="Username" required>
    </div>
    <div class="input-group">
        <i class="fa fa-envelope icon"></i>    
        <input type="email" name="email" placeholder="Email" required>
    </div>
    <div class="input-group">
        <i class="fa fa-lock icon"></i>
        <input type="password" id="password" name="password" placeholder="Password" required />
        <div type="button" class="toggle-password" onclick="togglePassword()">
            <i class="fa-solid fa-eye" id="eye-icon"></i>
        </div>
    </div>
    <div class="input-group">
        <i class="fa fa-lock icon"></i>
        <input type="password" id="con-password" name="Confirm-password" placeholder="Confirm Password" required />
        <div type="button" class="toggle-password" onclick="togglePasswords()">
            <i class="fa-solid fa-eye" id="eye-icons"></i>
        </div>
    </div>
    <button type="submit">Sign Up</button>
    <p>Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
    <p> By clicking "Sign Up" button, i expressly agree to CoreEfx AI <a href="{{ url_for('terms_of_service') }}">Terms of Service</a> and understand that my account information will be used according to CoreEfx AI 
    <a href="{{ url_for('privacy_policy') }}">Privacy Policy</a></p>
  </form>

  <script>
    function togglePassword() {
      const password = document.getElementById("password");
      const icon = document.getElementById("eye-icon");

      if (password.type === "password") {
        password.type = "text";
        icon.classList.remove("fa-eye");
        icon.classList.add("fa-eye-slash");
      } else {
        password.type = "password";
        icon.classList.remove("fa-eye-slash");
        icon.classList.add("fa-eye");
      }
    }
    function togglePasswords() {
      const password = document.getElementById("con-password");
      const icon = document.getElementById("eye-icons");

      if (password.type === "password") {
        password.type = "text";
        icon.classList.remove("fa-eye");
        icon.classList.add("fa-eye-slash");
      } else {
        password.type = "password";
        icon.classList.remove("fa-eye-slash");
        icon.classList.add("fa-eye");
      }
    }
  </script>
</body>
</html>
"""
# HTML Template for the Web Interface
# This string contains the full HTML structure and inline JavaScript for the front-end.
main_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoreEfx AI Health Advisor</title>
    <link rel="icon" type="image/png" href="{{ url_for('static', filename='images/brain.png') }}">
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {
            /* Light Mode Colors */
            --primary-color: #007bff; /* A professional blue */
            --secondary-color: #6c757d; /* Muted grey */
            --accent-color: #008000; /* Success green */
            --hover-accent: #28a428; /* green for hover */
            --background-light: #f8f9fa;
            --background-white: #ffffff;
            --text-dark: #343a40;
            --text-muted: #6c757d;
            --border-color: #e2e6ea;
            --shadow-light: rgba(0, 0, 0, 0.08);
            --transition-speed: 0.5s;
        }

        /* Dark Mode Colors */
        body.dark-mode {
            --primary-color: #6aabff; /* Lighter blue for dark mode */
            --secondary-color: #a0a8b1; /* Lighter grey for dark mode */
            --accent-color: #38b45a; /* Slightly lighter green for dark mode */
            --hover-accent: #2e8b43; /* hover green for dark mode */
            --background-light: #2c2f33; /* Darker background */
            --background-white: #36393f; /* Even darker background for cards/containers */
            --text-dark: #e0e0e0; /* Light text */
            --text-muted: #b0b0b0; /* Lighter muted text */
            --border-color: #4a4d52; /* Darker border */
            --shadow-light: rgba(0, 0, 0, 0.3); /* More pronounced shadow */
        }
        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: 'Open Sans', sans-serif; /* Professional, legible font */
            margin: 0;
            padding: 1px;
            height: 100%;
            background-color: var(--background-light);
            color: var(--text-dark);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            transition: background-color var(--transition-speed), color var(--transition-speed); /* Smooth transition for theme change */
           /* background-image: url("{{ url_for('static', filename='') }}"); /* Replace background.png with your image filename */
            background-size: cover; /* Common: make image cover the entire body, cropping if necessary */
            background-repeat: no-repeat; /* Prevent image from repeating */
            background-attachment: fixed; /* Keep background fixed when scrolling */
            background-position: center center; /* Center the background image */*/
    /* ------------------------------------------- */
            
        }

        .container {
            max-width:800px;
            min-height: calc(100dvh - 70px);
            margin: 0 auto;
            background-color: var(--background-white);
            border-radius: none;
            box-shadow: 0 8px 20px var(--shadow-light);
            padding: 80px 20px;
           
            box-sizing: border-box;
            transition: background-color var(--transition-speed), box-shadow var(--transition-speed); /* Smooth transition */
            
        }
        
        h3, h4 {
            font-family: 'Roboto', sans-serif; /* A more prominent font for headings */
            /*color: #008000;*/
            text-align: center;
            margin-bottom: 25px;
            font-weight: 700; /* Bolder headings */
            transition: color var(--transition-speed); /* Smooth transition */
        }
       

        h2 {
            font-size: 2em;
            margin-bottom: 10px;
            color: var(--accent-color);
            font:700;
            margin-left: 25px;
            transition: color var(--transition-speed); /* Smooth transition */
        }

        h3 {
            font-size: 1.8em;
            margin-top: 30px;
            color: var(--accent-color);
        }

        h4 {
            font-size: 1.4em;
            margin-top: 25px;
            color: var(--accent-color);
        }
        

        .intro-welcome{
            align-self: flex-start;
            background-color: var(--background-light);
            color: var(--text-dark);
            padding: 15px 18px;
            border-radius: 15px 15px 15px 0;
            max-width: 60%;
            text-align: left;
            font-weight: 500;
            word-wrap: break-word;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            font-size: 1em;
            transition: color var(--transition-speed); /* Smooth transition */
            margin-top: 40px;
            margin-bottom:-48px;
            
        }
        .intro-text { text-align: center; color: var(--text-muted); margin-bottom: 30px; font-size: 1.1em;  transition: color var(--transition-speed); /* Smooth transition */ }

        /* Tab Navigation Styles */
        .tabs {
            display: flex;
            justify-content: space-around;
            align-item: centre;
            width:100%;
            max-width:800px;/* matches container width */
        }

        .tab-button {
            background:none;
            border: none;
            flex: 1;
            padding: 8px 0;
            cursor: pointer;
            font-size: 14px;
            font-weight: 300;
            text-align: centre;
            display: flex;
            flex-direction: column;
            color: var(--text-dark);
            transition: color var(--transition-speed), border-bottom var(--transition-speed);
            position: relative;
            outline: none; /* Remove outline on focus for cleaner look */
        }

        .tab-button:hover .nav-icon {
            color: var(--hover-accent); /* Change text color to accent on hover */
            
        }
        

        .tab-button.active{
            font-weight: 1000;
        }
        
        .tab-button .nav-icon {
            font-size: 18px;
            margin-bottom: 6px;
            color: var(--text-dark);
            transition: color var(--transition-speed);
        }
        .tab-button.active .nav-icon {
          color: var(--accent-color); /* ✅ icon color when active */
        }
        
        
        /* App Header for Desktop (default) */
        .app-header {
            display: flex; /* Hidden by default on desktop */
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
            justify-content: space-between;
            align-items: center;
            max-width: 800px;
            background-color: var(--background-light);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            z-index:1000;
            position: fixed;
            padding: 5px 1px;
        }
        

        /* Tab Content Styles */
        .tab-content {
            display: none; /* Hidden by default */
            padding-top: 20px; /* Space from tabs */
        }

        .tab-content.active {
            display: block; /* Shown when active */
        }
       
        .audio-icon {
            margin-top: 10px;
        }
        .audio-icon .speak-btn {
            background: none; /* darker green button */
            color: var(--accent-color);
            border: none;
            padding: 10px 15px;
            margin-top: 10px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
        }
        .audio-icon .speak-btn:hover {
            color: var(--hover-accent); /* Slightly darker green for hover */
        }
        
        form {
            padding: 0;
            border-radius: 8px;
            box-shadow: none;
            background-color: transparent;
            margin-top: 140px; 
            top: 150px;
        }
        .input-with-icon {
           
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            background-color: var(--background-white);
            border: 1px solid var(--border-color);
            border-radius: 25px;
            padding: 8px 10px;
            box-shadow: none;
            width: 90%;
            max-width: 600px;
            
            transform: translateX(9%);
            
            transition: all 0.3s ease;
            z-index: 999; 
        }
        .input-with-icon:focus-within{
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 0.2rem rgb(0, 256, 55, 0.1);
            outline: none;
            background: white;
        }
        
        .input-with-icon textarea::-webkit-scrollbar {
          display: none;
        }
        .input-with-icon.active{
        background: white;
        }
        
        
        .input-with-icon textarea{
          flex: 1;
          border: none;
          outline: none;
          resize: none;
          font-size: 1em;
          font-family: 'Open Sans', sans-serif;
          color: var(--text-dark);
          background: transparent;
          padding: 8px;
          line-height: 1.4;
          overflow-y: auto;
          min-height: 40px;
          max-height: 160px;
          -ms-overflow-style: none;  /* IE, Edge */
          scrollbar-width: none;     /* Firefox */
        }
        .input-with-icon textarea:focus-within{
            color: black;
        }
        
        
        
        
        .icon-group {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-left: 8px;
        }
        
        /* feedback */
       
        .input-feed-icon {
           
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            background-color: var(--background-white);
            border: 1px solid var(--border-color);
            border-radius: 25px;
            padding: 8px 10px;
            box-shadow: none;
            width: 95%;
            
            margin-bottom:none;
            transition: all 0.3s ease;
            z-index: 999; 
        }
        .input-feed-icon:focus-within{
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 0.2rem rgb(0, 256, 55, 0.1);
            outline: none;
            background: white;
        }
        
        .input-feed-icon textarea::-webkit-scrollbar {
          display: none;
        }
        
        
        .input-feed-icon textarea{
          flex: 1;
          border: none;
          outline: none;
          resize: none;
          font-size: 1em;
          font-family: 'Open Sans', sans-serif;
          color: var(--text-dark);
          background: transparent;
          padding: 8px;
          line-height: 1.4;
          overflow-y: auto;
          min-height: 40px;
          max-height: 160px;
          -ms-overflow-style: none;  /* IE, Edge */
          scrollbar-width: none;     /* Firefox */
        }
        .input-feed-icon textarea:focus-within{
            color: black;
        }
        
        
        
        
        .icon-feed {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-left: 8px;
        }
        

        
        /* 🎙 Mic stays fixed at the right, vertically centered */
        .mic-btn {
            background: none;
            border: none;
            color: var(--accent-color);
            border-radius: 50%;
            width: 36px;
            height: 36px;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
          
        }
        
        .mic-btn:hover {
          color: var(--hover-accent); 
        }


        .circle-btn {
            background-color: var(--accent-color);
            border: none;
            color: white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .circle-btn:hover {
          background-color: var(--hover-accent);
          transform: scale(1.05);
        }
        
            
        




        hr {
            border: 0;
            height: 1px;
            background: var(--border-color);
            margin: 40px 0;
            transition: background var(--transition-speed); /* Smooth transition */
        }

        .results-section, .history-section, .feedback-form-section { /* Renamed feedback-section to feedback-form-section for clarity */
            margin-top: 40px;
            padding-top: 20px;
            
        }
        .chat-area {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: 12px;
        }

         .reply-me{
            align-self: flex-start;
            background-color: var(--background-light);
            color: var(--text-dark);
            padding: 15px 18px;
            border-radius: 15px 15px 15px 0;
            max-width: 60%;
            text-align: left;
            word-wrap: break-word;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            font-size: 1em;
            font-weight: 500;
            animation: fadeIn 1s ease-in;
         }
         .input-sympt{
            align-self: flex-end;
            background-color: var(--accent-color);
            color: white;
            padding: 15px 18px;
            border-radius: 15px 15px 0 15px;
            max-width: 60%;
            text-align: left;
            word-wrap: break-word;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            font-size: 1em;
            font-weight: 500;
            animation: fadeIn o.4s ease-in;
         }
         @keyframes fadeIn {
              from {opacity:0; transform: translateY(10px);}
              to {opacity:1; transform: translateY(0);}
         }
         
                        

        .results-section:first-of-type, .history-section:first-of-type, .feedback-form-section:first-of-type {
            margin-top: 0;
            padding-top: 0;
        }

        ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        li {
            background-color: var(--background-light);
            padding: 18px 20px;
            margin-bottom: 12px;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 1.05em;
            transition: background-color var(--transition-speed), box-shadow var(--transition-speed); /* Smooth transition */
        }

        li a {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 600;
            transition: color var(--transition-speed);
        }

        li a:hover {
            color: #0056b3;
            text-decoration: underline;
        }

        body.dark-mode li a:hover {
            color: #8bbdff; /* Lighter hover for dark mode links */
        }


        p {
            margin-top: 20px;
            color: var(--text-muted);
            text-align: center;
            font-size: 0.95em;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
            transition: color var(--transition-speed); /* Smooth transition */
        }

        .history-item {
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            transition: border-color var(--transition-speed); /* Smooth transition */
        }

        .history-item:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        .history-item p {
            text-align: left;
            margin: 5px 0;
            max-width: 100%;
            color: var(--text-dark);
            transition: color var(--transition-speed); /* Smooth transition */
        }

        .history-item .timestamp {
            font-size: 0.85em;
            color: var(--text-muted);
            text-align: left;
            display: block;
            margin-top: 10px;
            transition: color var(--transition-speed); /* Smooth transition */
        }
        /* NEW: Specific styling for the advice list and its items */
        .advice-list {
            list-style: disc; /* Use bullet points */
            margin-left: 25px; /* Indent the list */
            margin-top: 40px; /* Space from the "Advice:" heading */
            margin-bottom: 3px; /* Space below the entire advice list */
        }
        
        .advice-list li {
            background-color: transparent; /* No background for individual advice points */
            box-shadow: none; /* No shadow for individual advice points */
            padding: 0; /* Remove padding from individual advice points */
            margin-bottom: 8px; /* Space between each advice point */
            display: list-item; /* Ensure it behaves like a list item */
            align-items: flex-start; /* Align bullet points correctly */
            justify-content: flex-start;
            color: var(--text-dark); /* Ensure text color is correct */
            font-size: 1em; /* Adjust font size if needed */
        }
        
        .advice-list li:last-child {
            margin-bottom: 0; /* No bottom margin for the last advice point */
        }


        .feedback-message {
            text-align: center;
            color: var(--accent-color);
            font-weight: 600;
            margin-bottom: 20px;
            transition: color var(--transition-speed); /* Smooth transition */
        }

        #mapid {
            height: 450px;
            max-width: 100%;
            margin: 30px auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            /* Note: Leaflet map tiles are usually external and don't automatically theme. */
            /* You might need to use dark-themed map providers for full dark mode consistency. */
        }

        
        /* ✅ NAVBAR STYLES */
        .navbar {
          background: none !important;
          display: flex;
          justify-content: flex-end;
          align-items: center;
          padding: 0;
          box-shadow: none;
          position: relative;
          z-index: 1000;
          background-color: none !important;
          right: 20px;
         
          
        }
        
        .nav-toggle {
          background: none; 
          border: none; 
          color: var(--text-dark); 
          font-size: 1.5em; 
          cursor: pointer; 
          display: flex; 
          align-items: center; 
          font-weight: bold; 
        }
        
        .nav-toggle:hover {
          color: var(--hover-accent);
          transform: scale(1.05);
          transition: color 0.2s ease, transform 0.2s ease;
        }
        
       
        .nav-menu {
          display: none;
          flex-direction: column;
          position: absolute;
          
          top: 30px;
          right: -18px;
          
          
          background: var(--background-light);
          border-radius: 10px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
          padding: 10px 0;
          min-width: 200px;
          z-index: 1001;
        }
        
        .nav-menu a, .nav-menu button {
          text-align: Center;
          text-decoration: none;
          background: none;
          border: none;
          width: 100%;
          padding: 10px 2px;
          font-size: 1em;
          color: var(--text-dark);
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        
        .nav-menu a:hover, .nav-menu button:hover {
          background: var(--background-white);
          border-radius: 15px;
          color: var(--accent-color);
        }
        
        .nav-menu.show {
          display: flex;
          
        }
        
        /* Dropdown submenu inside navbar */
        .dropdown {
          position: relative;
          width: 100%;
        }
        
        .dropdown-content {
          display: none;
          flex-direction: column;
          background: var(--background-light);
          border-radius: 15px;
          margin-left: 6px;
          box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .logout-btn {
          display: inline-block !important; 
          margin-top: 10px;
          border-radius: 20px;
          
        }
        .mobile-logout{
          display: none !important;
        }
        
        
        
        
        .dropdown-content button {
          padding: 8px 20px;
          
        }
        
        .dropdown.show .dropdown-content {
          display: flex;
        }
        
        
        
        
        
        
        
        
        
        

        /* When the navigation is expanded */
        .tabs.active {
            display: flex; /* Show the tabs */
            flex-direction: column; /* Stack them vertically */
            max-width: 100%;
            margin-top: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05); /* Slight shadow for the dropdown */
            border-radius: 8px;
            background-color: var(--background-white); /* Match container background */
        }

        .tabs.active .tab-button {
            border-bottom: 1px solid var(--border-color); /* Separator for mobile tabs */
            width: 100%;
            text-align: left;
            padding: 15px 20px;
        }

        .tabs.active .tab-button:last-child {
            border-bottom: none;
        }
        
       
         /* Responsive adjustments */
        @media (max-width: 768px) {
            .footer-nav {
                height: 80px;
                padding bottom: 50px;
            }
            .container {
                
                width: 100%;
                min-width:100%;
                min-height: 100dvh; /* ✅ fills full device viewport height (modern and reliable) */
                margin: 0;
                padding: 80px 20px;
                box-sizing: border-box;
                background-color: var(--background-white);
                border-radius: 0;
                box-shadow: 0 8px 20px var(--shadow-light);
                transition: background-color var(--transition-speed), box-shadow var(--transition-speed);
            }
            .tab-button {
                font-size: 14px;
                padding: 10px 0;
            }
            .tab-button .nav-icon {
                font-size: 19px;
                margin-bottom:10px;
            }
            h2 {
                font-size: 1.6em;
                text-align:left;
                
            
            }
            h3 {
                font-size: 1.5em;
            }
            h4 {
                font-size: 1.2em;
            }
            .input-with-icon {
                bottom: 100px; /* keeps it above footer on mobile */
                width: 80%;
                padding: 10px 12px;
                border-radius: 20px;
            }
            .input-feed-icon{
                bottom: 300px; /* keeps it above footer on mobile */
                width: 90%;
                padding: 10px 12px;
                border-radius: 20px;
            }
                
            
            .input-with-icon textarea {
                font-size: 0.95em;
                padding: 8px 10px;
                min-height: 35px;
                max-height: 100px;
            }
            
            .mic-btn,
            .circle-btn {
                font-size: 18px;
            }

            
           
            .reply-me{
               max-width: 70%;
               font-size: 0.95em;
               padding: 12px 14px;
                
            }
            .input-sympt{
                max-width: 70%;
                font-size: 0.95em;
                padding: 12px 14px;
             }
            .intro-welcome{
                max-width: 70%;
                font-size: 0.95em;
                padding: 12px 14px;
            }
            
                
            
            
             /* Make tabs full width on smaller screens */
            .tab-button {
                flex-grow: 1; /* Make buttons take equal width */
                padding: 10px 15px;
                font-size: 0.9em;
            }
            
            li {
                padding: 15px;
                font-size: 1em;
                flex-direction: column;
                align-items: flex-start;
            }
            li a {
                margin-top: 5px;
            }
            
            /* Mobile-specific header layout */
            .app-header {
                display: flex; /* Activate flexbox for mobile */
                justify-content: space-between; /* Space out hamburger and settings */
                align-items: center; /* Vertically align them */
                padding: 10px 0; /* Adjust padding for mobile header */
                margin-bottom: 15px;
            }
            .logout-btn {
                display: none !important;
            }
            .mobile-logout {
                display: flex !important;
                
            }
        }

        @media (max-width: 480px) {
            body {
                padding: 0;
            }
            .footer-nav {
                height: 100px; /* taller footer on mobile */
            }
            .container {
                width: 150%;
                max-width: 100%;
                min-height: 100dvh; /* ✅ fills full device viewport height (modern and reliable) */
                margin: 0;
                padding: 80px 20px;
                box-sizing: border-box;
                background-color: var(--background-white);
                border-radius: 0;
                box-shadow: 0 8px 20px var(--shadow-light);
                transition: background-color var(--transition-speed), box-shadow var(--transition-speed);
               
            }
            
            h2 {
                font-size: 1.5em;
                text-align: left;
                
            }
           .input-with-icon {
                
                width: 80%;
                align-item: center;
                margin-right: 30%;
                border-radius: 18px;
                padding: 8px 10px;
                top: 30%;
           }
           .input-feed-icon {
                bottom: 120px;
                width: 90%;
                border-radius: 18px;
                padding: 8px 10px;
           }
            
           .input-with-icon textarea {
                font-size: 0.9em;
                min-height: 32px;
                max-height: 90px;
           }
            
            .mic-btn,
            .circle-btn {
                font-size: 17px;
            }
            
            .reply-me{
                max-width: 65%;
                font-size: 0.9em;
                padding: 10px 12px;
            }
             .input-sympt{
                max-width: 65%;
                font-size: 0.9em;
                padding: 10px 12px;
             }
             .intro-welcome{
                max-width: 65%;
                font-size: 0.9em;
                padding: 10px 12px;
            }
            
            
            .logout-btn {
                display: none !important;
            }
            .mobile-logout {
                display: flex !important;
               
            }
            #mapid {
                height: 300px;
            }
            .tabs {
                flex-wrap: wrap; /* Allow tabs to wrap on smaller screens */
                padding-right: 0px;
            }
            .tab-button {
                padding: 12px 0; /* Full width for tabs on very small screens */
                font-size: 13px;
            }
            /* Adjust top-right controls for very small mobile */
            .top-right-controls {
                top: 10px;
                right: 10px;
            }
            .tab-button .nav-icon {
                font-size: 20px;
                margin-bottom:
            }
            
           
             main, .container {
                padding-bottom: 10px; /* extra space for mobile footer */
               
             }
            
        }
        /* Footer container */
        .footer-nav {
          position: fixed;
          bottom: 0;
          left: 0;
          width: 100%;
          background: transparent; /* no border here */
          z-index: 1000;
          display: flex;
          height:70px;
          justify-content: center; /* center the container */
        }
        
        /* The actual container inside */
        .footer-container {
          width: 800px;              /* match your container width */
          max-width: 100%;           /* responsive */
          background: var(--background-light);
          border-top: 1px solid var(--shadow-light); /* border only inside */
          box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.1);
    
        }
        
        
        /* Make sure content isn’t hidden behind footer */
        main, .container {
          padding-bottom: 100px;
        }   
    </style>
</head>
<body>
    <div class="container">
        <!-- New app-header for mobile alignment -->
        <header class="app-header">
            <h2><i class="fa-solid fa-brain fa-icon-large"></i>CoreEfx AI</h2>
            <!-- ✅ NEW NAVBAR -->
            <nav class="navbar">
              <button class="nav-toggle" id="navToggle">
                <i class="fa-solid fa-ellipsis-vertical"></i>
              </button>
            
              <div class="nav-menu" id="navMenu">
                <div class="dropdown">
                  <button class="dropdown-btn"><i class="fa fa-gear"></i> Settings ▾</button>
                  <div class="dropdown-content">
                    <button class="theme-toggle" onclick="toggleTheme(); toggleNavMenu();">
                      <i class="fa fa-moon"></i>
                      <span>Dark Mode</span>
                    </button>
                    <button class="theme-toggle" onclick="toggleTheme(); toggleNavMenu();">
                        <i class="fa fa-desktop"></i>
                        <span>System Mode</span>
                    </button>
    
                    <button class="mobile-logout" onclick="window.location='{{ url_for('logout') }}'">
                      <i class="fa fa-sign-out-alt"></i> 
                      <span>Logout</span>
                    </button> 
                  </div>
                </div>
                <a href="{{ url_for('privacy_policy') }}"><i class="fa fa-lock"></i> Privacy Policy</a>
                <a href="{{ url_for('terms_of_service') }}"><i class="fa fa-file-alt"></i> Terms of Service</a>
                
                <button class="logout-btn" onclick="window.location='{{ url_for('logout') }}'">
                    <i class="fa fa-sign-out-alt"></i> 
                    <span>Logout</span>
                </button>
              </div>
            </nav>

        </header>
            
        <div id="home" class="tab-content active">
           

            <div class="intro-welcome"><b>Welcome {{ current_user.username }}, as your AI Health Advisor, I provide initial symptom guidance & help you find nearby hospitals. Please remember, a medical professional diagnosis is essential.</b></div>
            
            

            {% if user_message or result %}
            
            <div class="results-section">
                 <div class="chat-area">  <!-- ✅ New wrapper -->    
                    {% if user_message %}
                        <div class="input-sympt">
                            <b>{{ user_message }}</b>
                        </div>     
                    {% endif %}
                
                    {% if result %}
                       
                        {% for item in result %}
                        <div class="reply-me">
                            <b>{{ item }}</b>
                        </div>
                           
                        {% endfor %}
                 </div>   
                    {% if audio_file %}
                    <div class="audio-icon">
                        <button onclick="playAdvice()" class="speak-btn">
                            <i class="fa-solid fa-volume-low"></i>
                        </button>
                        <audio id="adviceAudio" style="display:none;">
                            <source src="{{ audio_file }}" type="audio/mpeg">
                        </audio>
                    </div>
                    {% endif %}
            
                    <h4>Here are nearby hospitals that may help, based on your location:</h4>
                    <ul>
                    {% for hospital in hospitals %}
                        <li><b>
                            {{ hospital.name }} ({{ hospital.city.title() }})</b>
                            {% if hospital.url %}
                                <a href="{{ hospital.url }}" target="_blank" title="{{ hospital.url }}">More Info</a>
                            {% endif %}
                        
                        </li>
                    {% endfor %}
                    </ul>
                    <div id="mapid"></div>
                {% endif %}
            </div>
            
            {% endif %}

            
            <form method="POST">
                
                <div class="input-with-icon">
                    <textarea type="text" id="symptoms" name="symptoms" rows="5"  required class="symptom-input" placeholder="What symptoms are you experiencing today? e.g., I have a fever, chills, and a headache."></textarea>
                    <div class ="icon-group">
                        <button type="button" class="mic-btn" onclick="startDictation('symptoms')"><i class="fa-solid fa-microphone"></i></button>
                        <button type="submit" class="circle-btn"><i class="fa-solid fa-paper-plane"></i></button>
                    </div>
                </div>
                
                <input type="hidden" name="lat" id="lat">
                <input type="hidden" name="lon" id="lon">
            </form>
        </div>

        <div id="history" class="tab-content">
            <div class="history-section">
                <h3>Your Health Report:</h3>
                {% if history_reports %}
                    <ul>
                    {% for report in history_reports %}
                        <li class="history-item">
                            <div>
                                <p><strong>Symptoms:</strong> {{ report.input_text }}</p>
                                <p><strong>Advice:</strong></p> {# Start a new paragraph for the "Advice:" label #}
                                <ul> {# Use an unordered list for multiple advice points #}
                                {% if '\n' in report.result %} {# Check if there are multiple lines #}
                                    {% for advice_point in report.result.split('\n') %}
                                        {% if advice_point.strip() %} {# Ensure no empty list items from multiple newlines #}
                                            <li class="single-advice-item">{{ advice_point.strip() }}</li>
                                        {% endif %}
                                    {% endfor %}
                                {% else %}
                                    <li class="single-advice-item">{{ report.result }}</li> {# If only one advice, display it #}
                                {% endif %}
                                </ul>
                                <div><span class="timestamp">{{ report.timestamp.strftime('%d-%m-%Y %H:%M') }}</span></div>
                            </div>
                            
                            
                         </li>
                    {% endfor %}
                    </ul>
                {% else %}
                    <b><p class="intro-text">No past consultations yet. Submit your symptoms on the Home tab to see your history here!</p></b>
                {% endif %}
            </div>
        </div>    

        <div id="feedback" class="tab-content">
            <div class="feedback-form-section">
                <h3>We Value Your Feedback!</h3>
                <p class="intro-text"><b>Your input helps us improve. Please share your thoughts on your experience.</b></p>
                {% if feedback_message %}
                    <p class="feedback-message">{{ feedback_message }}</p>
                {% endif %}
                
                
                <form method="POST">
                    <label for="feedback_message" class="label-tittle"><b>Your Feedback:</b></label><br><br>
                        <div class="input-feed-icon">
                            <textarea type="text" id="feedback_message" name="feedback_message"  required class="feedback_message" rows="5" class="feedback-textarea"  placeholder="e.g., The advice was helpful, but the map didn't load..."></textarea> 
                            <div class ="icon-group">
                                <button type="button" class="mic-btn" onclick="startDictation('feedback_message')"><i class="fa-solid fa-microphone"></i></button>
                                <button type="submit" class="circle-btn"><i class="fa-solid fa-paper-plane"></i></button>
                            </div>
                        </div>
                </div>
                </form>
            </div>
        </div>
    <footer class="footer-nav">
        <div class="footer-container">
            <div class="tabs" id="mainTabs">
                <button class="tab-button active" onclick="openTab(event, 'home')">
                    <i class="fa-solid fa-brain fa-icon-large nav-icon"></i>
                    <span>Home</span>
                </button>
                <button class="tab-button" onclick="openTab(event, 'feedback')">
                    <i class="fas fa-comment-alt nav-icon"></i>
                    <span>Feedback</span>
                </button>
                <button class="tab-button" onclick="openTab(event, 'history')"> 
                    <i class="fas fa-history nav-icon"></i>
                    <span>History</span>
                </button>   
            </div>
        </div>
    </footer>    
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  

    <script>
        let map;
        
        
        document.querySelectorAll('textarea').forEach(textarea => {
          textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
          });
        });


        
        // NEW JAVASCRIPT FUNCTION FOR MOBILE NAV TOGGLE -- MOVED HERE
        const navToggle = document.getElementById('navToggle');
        const navMenu = document.getElementById('navMenu');
        const dropdownBtn = document.querySelector('.dropdown-btn');
        const dropdown = document.querySelector('.dropdown');
        
        navToggle.addEventListener('click', () => {
          navMenu.classList.toggle('show');
        });
        
        dropdownBtn.addEventListener('click', (e) => {
           e.stopPropagation();
           dropdown.classList.toggle('show');
        });
        
        document.addEventListener('click', (event) => {
          if (!navMenu.contains(event.target) && !navToggle.contains(event.target)) {
            navMenu.classList.remove('show');
            dropdown.classList.remove('show');
          }
        });
        
        function toggleNavMenu() {
          navMenu.classList.remove('show');
          dropdown.classList.remove('show');
        }


      

        // Theme Toggle Functionality -- MOVED HERE (or keep inside DOMContentLoaded if it strictly depends on DOM being ready)


        // THEME CONTROL
        const body = document.body;
        const themeBtn = document.querySelector('.theme-toggle i');
        const themeText = document.querySelector('.theme-toggle span');
        
        let systemModeActive = true;
        const systemQuery = window.matchMedia('(prefers-color-scheme: dark)');
        
        // Apply selected theme
        function applyTheme(mode) {
          body.classList.remove('dark-mode');
          
          if (mode === 'dark') {
            body.classList.add('dark-mode');
            themeBtn.className = 'fa fa-sun';
            themeText.textContent = 'Light Mode';
          } else if (mode === 'light') {
            themeBtn.className = 'fa fa-moon';
            themeText.textContent = 'Dark Mode';
          } else {
            // System Mode UI indicator
            themeBtn.className = 'fa fa-desktop';
            themeText.textContent = 'System Mode';
          }
        }
        
        // Force manual theme
        function setTheme(mode) {
          localStorage.setItem('theme', mode);
          applyTheme(mode);
          systemModeActive = false;
          systemQuery.removeEventListener('change', handleSystemChange);
        }
        
        // Switch between dark/light/system manually
        function toggleTheme() {
          const currentMode = localStorage.getItem('theme') || 'system';
        
          if (currentMode === 'light') {
            setTheme('dark');
          } else if (currentMode === 'dark') {
            setSystemMode(); // Switch to system mode next
          } else {
            setTheme('light');
          }
        }
        
        // Follow system automatically
        function setSystemMode() {
          localStorage.setItem('theme', 'system');
          applyTheme('system');
          systemModeActive = true;
        
          // Apply system’s current theme immediately
          applyTheme(systemQuery.matches ? 'dark' : 'light');
        
          // Update dynamically if system theme changes
          systemQuery.addEventListener('change', handleSystemChange);
        }
        
        function handleSystemChange(e) {
          if (systemModeActive) {
            applyTheme(e.matches ? 'dark' : 'light');
          }
        }
        
        // On page load
        document.addEventListener('DOMContentLoaded', () => {
          const saved = localStorage.getItem('theme') || 'system';
          
          if (saved === 'dark') {
            applyTheme('dark');
          } else if (saved === 'light') {
            applyTheme('light');
          } else {
            setSystemMode();
          }
        });
        
        function playAdvice() {
            let audio = document.getElementById("adviceAudio");
            audio.play();
        }
            
        // ✅ NEW: Voice Input (Speech-to-Text)
        function startDictation(targetId) {
          if ('webkitSpeechRecognition' in window) {
            let recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = "en-US";
            recognition.start();
        
            recognition.onresult = function (e) {
              const transcript = e.results[0][0].transcript;
              document.getElementById(targetId).value = transcript;
              recognition.stop();
            };
        
            recognition.onerror = function (event) {
              console.warn("Speech recognition error:", event.error);
              recognition.stop();
            };
          } else {
            alert("Your browser doesn't support voice input. Try Chrome.");
          }
        }


                


        // Geolocation functions (keep these outside DOMContentLoaded as they are callable by themselves)
        function getLocation() {
            console.log("Attempting to get location...");
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(showPosition, showError, {timeout: 10000});
            } else {
                console.log("Geolocation is not supported by this browser.");
                // Using a custom message box instead of alert()
                const messageBox = document.createElement('div');
                messageBox.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background-color: #e2e3e5;
                    color: #383d41;
                    border: 1px solid #d6d8db;
                    border-radius: 5px;
                    padding: 20px;
                    z-index: 9999;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    font-family: Arial, sans-serif;
                    text-align: center;
                `;
                messageBox.innerHTML = `
                    <p>Geolocation is not supported by this browser. Cannot provide nearby hospitals.</p>
                    <button onclick="this.parentNode.remove()" style="
                        background-color: #6c757d;
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        border-radius: 4px;
                        cursor: pointer;
                        margin-top: 10px;
                    ">OK</button>
                `;
                document.body.appendChild(messageBox);
            }
        }

        function showPosition(position) {
            console.log("Location obtained successfully!");
            document.getElementById("lat").value = position.coords.latitude;
            document.getElementById("lon").value = position.coords.longitude;
            console.log("Lat:", position.coords.latitude, "Lon:", position.coords.longitude);

            const userLat = position.coords.latitude;
            const userLon = position.coords.longitude;
            const hospitalsData = {{ hospitals | tojson }};

            if (hospitalsData && hospitalsData.length > 0 && document.getElementById('mapid')) {
                 initializeMap(userLat, userLon, hospitalsData);
            } else {
                console.log("Map not initialized: No hospitals to display or map container not found.");
            }
        }

        function showError(error) {
            console.error("Geolocation error object:", error);
            let errorMessage = "An unknown geolocation error occurred.";
            switch(error.code) {
                case error.PERMISSION_DENIED:
                    errorMessage = "User denied the request for Geolocation. Please allow location access in your browser settings to find nearby hospitals.";
                    console.error("User denied the request for Geolocation.");
                    break;
                case error.POSITION_UNAVAILABLE:
                    errorMessage = "Location information is unavailable. This may be due to network issues or your device's settings.";
                    console.error("Location information is unavailable.");
                    break;
                case error.TIMEOUT:
                    errorMessage = "The request to get user location timed out. Please check your internet connection or try again.";
                    console.error("The request to get user location timed out.");
                    break;
                case error.UNKNOWN_ERROR:
                    errorMessage = "An unknown error occurred while getting your location.";
                    console.error("An unknown error occurred.");
                    break;
            }
            console.error(errorMessage);
            document.getElementById("lat").value = "";
            document.getElementById("lon").value = "";
            
            // Display custom message box for geolocation errors
            const messageBox = document.createElement('div');
            messageBox.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
                padding: 20px;
                z-index: 9999;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                font-family: Arial, sans-serif;
                text-align: center;
            `;
            messageBox.innerHTML = `
                <p>${errorMessage}</p>
                <button onclick="this.parentNode.remove()" style="
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-top: 10px;
                ">OK</button>
            `;
            document.body.appendChild(messageBox);
        }

        function initializeMap(userLat, userLon, hospitalsData) {
            if (map && map.remove) {
                map.remove();
            }

            const mapDiv = document.getElementById('mapid');
            if (!mapDiv) {
                console.error("Map container #mapid not found. Cannot initialize map.");
                return;
            }

            map = L.map('mapid').setView([userLat, userLon], 12);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 100,
                //attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);

            L.marker([userLat, userLon]).addTo(map)
                .bindPopup("<b>Your Approximate Location</b>").openPopup();

            hospitalsData.forEach(function(hospital) {
                L.marker([hospital.lat, hospital.lon]).addTo(map)
                    .bindPopup(`<b>${hospital.name}</b><br>${hospital.city.charAt(0).toUpperCase() + hospital.city.slice(1)}<br><a href="${hospital.url}" target="_blank">More Info</a>`);
            });
        }

        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;

            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }

            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }

            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
            // --- END OF KEY PART ---

            if (tabName === 'home' && map) {
                setTimeout(() => {
                    map.invalidateSize();
                    if (document.getElementById("lat").value && document.getElementById("lon").value) {
                         const userLat = parseFloat(document.getElementById("lat").value);
                         const userLon = parseFloat(document.getElementById("lon").value);
                         const hospitalsData = {{ hospitals | tojson }};
                         initializeMap(userLat, userLon, hospitalsData);
                    }
                }, 0);
            }
        }

       
        // All DOM-related initializations go here
        document.addEventListener('DOMContentLoaded', () => {
            console.log("DOM content loaded for initial setup.");

           



            // Get location on page load for initial setup, even if not displayed immediately
            getLocation();

            // Ensure the correct tab is active on initial load (defaults to 'home') - if dynamically setting
            // openTab(null, 'home'); // Call this if you want to explicitly ensure 'home' is active
        });
        
    </script>
</body>
</html>
"""

#MIN_HYBRID_THRESHOLD = 5 # more lenient

@app.route("/", methods=["GET", "POST"])
def welcome():
    if current_user.is_authenticated:
        # If logged in, show welcome first, then go to home
        return render_template_string(welcome_template.replace(
            '</body>',
            """<script>
                setTimeout(() => {
                    window.location.href = '/home';
                }, 8000); // 8 seconds delay
            </script></body>"""
        ))
    else:
        # If not logged in, show welcome first, then go to login
        return render_template_string(welcome_template.replace(
            '</body>',
            """<script>
                setTimeout(() => {
                    window.location.href = '/login';
                }, 8000); // 8 seconds delay
            </script></body>"""
        ))
    

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, name=name, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template_string(signup_template)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            db.session.add(UserActivity(user_id=user.id, action="login"))
            db.session.commit()
            return redirect(url_for("index"))
        else:
            return "Invalid login."
    return render_template_string(login_template)

@app.route("/logout")
@login_required
def logout():
    db.session.add(UserActivity(user_id=current_user.id, action="logout"))
    db.session.commit()
    logout_user()
    return redirect(url_for("login"))

@app.route('/privacy_policy')
def privacy_policy():
    return render_template("privacy_policy.html")

@app.route('/terms_of_service')
def terms_of_service():
    return render_template('terms_of_service.html')



@app.route("/home", methods=["GET", "POST"])
@login_required
def index():
    user_input = ""  # ✅ Initialize to avoid UnboundLocalError

    result = []
    hospitals = []
    feedback_message = None  # To display confirmation after feedback submission
    audio_file = "" #default

    # Fetch history on GET request (initial page load) or after POST
    # Order by timestamp descending to show most recent first
    history_reports = SymptomReport.query.filter_by(user_id=current_user.id).order_by(
        SymptomReport.timestamp.desc()
    ).all()


    if request.method == "POST":
        # Check if it's a symptom submission or feedback submission
        if "symptoms" in request.form:
            user_input = request.form.get("symptoms", "").strip()
            lat_str = request.form.get("lat")
            lon_str = request.form.get("lon")

            # ✅ Require at least 2 words
            if not has_medical_relevance(user_input):
                msg = "I could not identify any symptoms. Please describe your health condition."
                result = [msg]
                audio_file = generate_audio(msg) or ""

            else:
                # ✅ 1. Get ML predictions (top 3)
                ml_results = ml_predict_condition(user_input, top_n=5, threshold=0.0)

                # ✅ 2. Get keyword matches
                keyword_matches = check_symptoms(user_input, min_score_threshold=1, top_n=5)

                # ✅ 3. Merge into one hybrid dictionary
                hybrid_results = {}
                for cond_name, prob in ml_results:
                    hybrid_results[cond_name] = {"ml_conf": prob, "spacy_score": 0}

                for cond_name, score in keyword_matches:
                    if cond_name in hybrid_results:
                        hybrid_results[cond_name]["spacy_score"] = score
                    else:
                        hybrid_results[cond_name] = {"ml_conf": 0, "spacy_score": score}


                # ✅ Compute hybrid score for ranking
                ranked_conditions = []
                for cond_name, vals in hybrid_results.items():
                    ml_conf = vals["ml_conf"]
                    spacy_score = vals["spacy_score"]
                    hybrid_score = (ml_conf * 100) + (spacy_score * 10)
                    ranked_conditions.append((cond_name, hybrid_score))

                # ✅ Sort by hybrid_score (highest first)
                ranked_conditions.sort(key=lambda x: x[1], reverse=True)

                # ✅ Pick top 3 conditions above a minimal threshold
                valid_conditions = [c for c, score in ranked_conditions if score >= 6][:5]

                # ✅ Build ONLY advice
                if valid_conditions:
                    advice_texts = []

                    for cond_name in  valid_conditions:
                        cond_key = cond_name.strip().lower()
                        if cond_key in symptom_data:
                            # ✅ Get just the advice sentence
                            advice = symptom_data[cond_key]["advice"]
                            result.append(advice)  # ✅ only advice, no condition name
                            advice_texts.append(advice)  # for audio
                        #else:
                            #print(f"⚠️ No advice found for: {cond_key}")

                    audio_file = generate_audio(" ".join(advice_texts)) or ""

                else:
                    msg = "\nI could not identify your condition. Please consult a doctor."
                    result = [msg]
                    audio_file = generate_audio(msg) or ""


            location_text = "N/A"
            user_lat, user_lon = None, None

            if lat_str and lon_str:
                try:
                    user_lat = float(lat_str)
                    user_lon = float(lon_str)
                    location_text = f"{user_lat},{user_lon}"
                    hospitals = find_nearby_hospitals(user_lat, user_lon, radius_km=50)
                    if not hospitals:
                        # If no hospitals are found within default radius, try a larger radius
                        hospitals = find_nearby_hospitals(user_lat, user_lon, radius_km=100) # Increased radius
                        if not hospitals:
                            result.append("\nNo nearby hospitals found within 200km. Please consider widening your search or manually looking for facilities.")
                        else:
                            result.append("\nWe found some hospitals, but they might be a bit further away. Check the map for details.")
                except ValueError:
                    result.append("\nInvalid location data received. Cannot find nearby hospitals.")
            else:
                result.append("\nYour location was not provided or could not be determined. Cannot find nearby hospitals.")

            # Save the symptom report to the database
            new_report = SymptomReport(
                user_id = current_user.id,
                input_text=user_input,
                location=location_text,
                result=" ".join(result) # Join list of advice strings into a single string
            )
            db.session.add(new_report)
            db.session.commit()

            # Refresh history after a new submission
            history_reports = SymptomReport.query.filter_by(user_id=current_user.id).order_by(
                SymptomReport.timestamp.desc()
            ).all()

            # This is important to ensure the map loads correctly after a POST request
            # if user_lat and user_lon:
            #     # No direct call to initializeMap needed here, the JS onload will handle it
            #     # by checking for `hospitals` data passed in render_template_string.
            #     pass # Handled by the JavaScript

        elif "feedback_message" in request.form:
            feedback_content = request.form["feedback_message"]
            if feedback_content:
                new_feedback = Feedback(message=feedback_content)
                db.session.add(new_feedback)
                db.session.commit()
                feedback_message = "Thank you for your feedback!"
            else:
                feedback_message = "Feedback message cannot be empty."

    safe_result = result if isinstance(result, list) else []
    safe_audio = audio_file if isinstance(audio_file, str) else ""
    safe_user = current_user if current_user else ""

    return render_template_string(main_template,result=safe_result, hospitals=hospitals, history_reports=history_reports, feedback_message=feedback_message,   audio_file=safe_audio, current_user=safe_user,  user_message=user_input)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
