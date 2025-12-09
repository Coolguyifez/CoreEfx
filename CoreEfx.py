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
    input_text = db.Column(db.Text)  # The raw text input from the user (symptoms)
    location =  db.Column(db.String(100))  # User's approximate location (latitude,longitude string)
    result = db.Column(db.Text)  # The health advice/diagnosis provided by the system (increased length)
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
            Hospital(name="Lagos State University Teaching Hospital (LASUTH)", city="lagos", lat=6.59047449787585,
                     lon=3.3422608588498037,
                     url="https://lasuth.org.ng/"),
            Hospital(name="University of Port Harcourt Teaching Hospital", city="East-West Road, PH",
                     lat=4.90053, lon=6.92877,
                     url="https://www.google.com/search?q=upth+port+harcourt&rlz"),
            Hospital(name="Rivers State University Teaching Hospital", city="Harley Street, Old GRA port harcourt",
                     lat=4.77999, lon=7.01429,
                     url="https://www.google.com/search?q=upth+port+harcourt&rlz"),
            Hospital(name="Professor Kelsey Harrison Hospital", city="11 Emenike Street, Mile 1, Diobu, Port harcourt",
                     lat=4.79096, lon=6.99437,
                     url="https://www.google.com/search?q=upth+port+harcourt&rlz"),
            Hospital(name="Rivers State Dental and Maxillofacial Hospital", city="Garrison, Port Harcourt",
                     lat=4.80552, lon=7.00923,
                     url="https://www.google.com/search?q=upth+port+harcourt&rlz"),
            Hospital(name="Obio Cottage Hospital",
                     city="near Trans-Amadi Industrial Layout Road, Rumuola, Port Harcourt", lat=4.83444, lon=7.03362,
                     url="https://farepharm.com/hospital/313"),
            Hospital(name="Portharcourt Government House Clinic", city="Forces Ave, Old GRA, Port Harcourt 500101",
                     lat=4.77454, lon=7.01643,
                     url="https://www.google.com/search?q=upth+port+harcourt&rlz"),
            Hospital(name="University of Abuja Specialist Hospital", city="abuja", lat=8.965031007601917,
                     lon=7.064360769741204,
                     url="https://www.google.com/search?q=university+of+abuja+specialist+hospital+gwagwalada&sca"),

            # Hospitals near Delta State,Rivers State, FCT, Lagos (specifically added for your location context)
            Hospital(name="Central Hospital - Warri Emergency Unit", city=" Mabiaku Rd, Warri-Ogunu Rd, Warri",
                     lat=5.51702, lon=5.73447,
                     url="https://www.google.com/search?q=central+hospital+warri&sca"),
            Hospital(name="Delta State University Teaching Hospital (DELSUTH)", city="Oghara", lat=5.9602250750294345,
                     lon=5.702942682383533,
                     url="https://www.google.com/search?q=delta+state+university+teaching+hospital&sca"),
            # Located in Oghara
            Hospital(name="Federal Medical Centre (FMC)", city=" Isieke-Asaba", lat=6.2121256496757615,
                     lon=6.7122751952742465,
                     url="https://www.google.com/search?q=Federal+Medical+Centre+Asaba&sca"),
            Hospital(name="University of Benin Teaching Hospital (UBTH)",
                     city=" Benin Lagos Express Road, Uselu, Benin City", lat=6.3903335466504725, lon=5.611826787114936,
                     url="https://www.google.com/search?q=ubth&sca"),
            Hospital(name="Federal Medical Centre Yenagoa", city="Hospital Rd, Yenagoa", lat=4.937291400004392,
                     lon=6.266638050617462,
                     url="https://www.google.com/search?q=federal+medical+centre+yenagoa&sca"),
            Hospital(name="Divine Grace Hospital Warri", city="2 Rubber Crescent, Warri ", lat=5.568987307199267,
                     lon=5.767263095268969,
                     url="https://www.bing.com/search?q=divine_grace_hospital%2ceffurun.delta_state&qs"),
            Hospital(name="Lily Hospitals Ltd. Warri", city="6 BrisibeLane, Deco Rd, off Etuwewe Road, Warri",
                     lat=5.526185526750632, lon=5.765111210609841,
                     url="https://lilyhospitals.com/"),
            Hospital(name="Asaba Specialist Hospital", city="GRA Phase I, Okpanam, Asaba", lat=6.234913709017079,
                     lon=6.685982039450925, url="https://asabaspecialisthospital.org/"),
            Hospital(name="Ughelli Central Hospital", city="Oteri, Ughelli", lat=5.49428, lon=5.99526,
                     url="https://www.cybo.com/NG-biz/ughelli-central-hospital"),
            Hospital(name="Sapele Central Hospital", city="Market Rd,Sapele", lat=5.90041, lon=5.68115,
                     url="https://www.cybo.com/NG-biz/sapele-central-hospital"),
            Hospital(name="Sapele Primary Health Centre", city="Market Rd,Sapele", lat=5.90041, lon=5.68115,
                     url="https://www.africabizinfo.com/NG/sapele-primary-health-centre"),
            Hospital(name=" Agbor Central Hospital", city="Lagos-Asaba Rd, Umutu-Aliagwai Agbor Rd, Agbor", lat=6.25575,
                     lon=6.18205, url="https://www.cybo.com/NG-biz/central-hospital-agbor"),
            Hospital(name="Oleh Central Hospital", city=" Oleh-Okpari-Warri Rd, Oleh", lat=5.47793, lon=6.20428,
                     url="https://www.cybo.com/NG-biz/central-hospital-oleh"),
            Hospital(name="Boji Boji Owa Primary Health Centre", city="Boji Boji, Agbor", lat=6.25396, lon=6.21760,
                     url="https://www.cybo.com/NG-biz/boji-boji-owa-primary-health-centre"),
            Hospital(name="SECHO Specialist Hospital", city="Okobi St, Boji Boji-Agbor", lat=6.25123, lon=6.19940,
                     url="https://www.cybo.com/NG-biz/secho-specialist-hospital"),
            Hospital(name=" Agbor General Hospital", city="Boji Boji-Agbor", lat=6.24236, lon=6.20610,
                     url="https://www.cybo.com/NG-biz/general-hospital_170c"),
            Hospital(name="Boji-Boji Agbor Primary Health Centre", city="Urubisi st,Agbor", lat=6.26338, lon=6.18474,
                     url="https://www.cybo.com/NG-biz/boji-boji-agbor-primary-health-centre"),
            Hospital(name="Onicha-Olona General Hospital ", city=" Atuma Rd, Onicha-Olona ", lat=6.36676, lon=6.56661,
                     url="https://www.africabizinfo.com/NG/onicha-olone-general-hospital"),
            Hospital(name="Onicha-Uku Primary Health Centre", city="Ugbodu Rd, Onicha-Uku", lat=6.37447, lon=6.46181,
                     url="https://www.africabizinfo.com/NG/onicha-uku-primary-health-centre"),
            Hospital(name="Issele-Uku Primary Health Centre", city="otolokpo Rd, Issele-Uku", lat=6.31878, lon=6.47624,
                     url="https://www.cybo.com/NG-biz/issele-uku-primary-health-centre"),
            Hospital(name="Ogwashi uku General hospital", city="Agidase, Ogwashi-Ukwu", lat=618167, lon=6.53025,
                     url="https://branches.com.ng/branch-detail/Hospitals-and-Clinics-in-Nigeria-General-Hospital-Ogwashi-Uku-Delta"),
            Hospital(name="Ubulu-Uku Health Centre", city="Ubulu-Uku", lat=6.23348, lon=6.44983,
                     url="https://branches.com.ng/branch-detail/Hospitals-and-Clinics-in-Nigeria-Ubulu-Uku-Primary-Health-Centre-Delta"),
            Hospital(name="Isheagu Government Hospital", city=" Ewulu/Ishagu, Isheagu", lat=6.03929, lon=6.54674,
                     url="https://thehospitalbook.com/isheagu-government-hospital/"),
            Hospital(name="Bomadi General Hospital", city="Bomadi", lat=5.16843, lon=5.91396,
                     url="https://branches.com.ng/branch-detail/Hospitals-and-Clinics-in-Nigeria-Bomadi-General-Hospital-Bomadi-Delta"),
            Hospital(name="Government Hospital Ehwerhe", city="Ehwerhe- Agbarho", lat=5.58586, lon=5.86606,
                     url="https://www.cybo.com/NG-biz/government-hospital-ehwerhe-agbarho"),
            Hospital(name="Orogun General Hospital", city="Ogor-otagba Rd, Ejeba, Orogun", lat=5.63683, lon=615289,
                     url=""),
            Hospital(name="Burutu General Hospital", city="Burutu", lat=5.35024, lon=5.51655, url=""),
            Hospital(name="Focados Government Hospital", city="Focados", lat=5.35884, lon=5.43361, url=""),
            Hospital(name="Cottage Hospital", city="Ogulagha", lat=5.35136, lon=5.34302, url=""),
            Hospital(name="Ojobo Government Hospital", city="Bolou-Ojobo", lat=5.03185, lon=5.66679, url=""),
            Hospital(name="Kiagbodo Government Hospital", city="Akugbene", lat=5.24324, lon=5.83601, url=""),
            Hospital(name="Eku Baptist Hospital", city="Igun Watershed, Eku", lat=5.75175, lon=5.99549, url=""),
            Hospital(name="Isokolo General Hospital",
                     city="Orhokpor Orokpor Rd, Ughelli-Isokolo Rd, Agbon VII, Isokolo", lat=5.59122, lon=5.98717,
                     url=""),
            Hospital(name="Isiokolo Health Centre", city="Isokoko-Egume Rd, Isokolo", lat=5.60097, lon=6.00240, url=""),
            Hospital(name="Abraka Central Hospital", city="Akure-Abedei Rd, Abraka", lat=5.78942, lon=6.10226, url=""),
            Hospital(name="Cottage Hospital Erhoike", city="Obajere-Orokpor Rd, Isokolo", lat=5.64449, lon=6.03590,
                     url=""),
            Hospital(name="Erhoike Cottage Hospital", city="Eko-Obiaruku Rd, Oria", lat=5.76508, lon=6.04901, url=""),
            Hospital(name="Abraka Health Centre", city=" Eko-Obiaruku Rd, Oria", lat=5.76678, lon=6.04982, url=""),
            Hospital(name="Oghara General Hospital", city="Oghara-Ajagbodudu Rd, Oghara", lat=5.93847, lon=5.67954,
                     url=""),
            Hospital(name="Great Land Hospital", city="College Rd, Mosogar", lat=5.90345, lon=5.73071, url=""),
            Hospital(name="Ugbevwe-Jesse Government Hospital", city="Ajavwuni Ugbevwe", lat=5.03185, lon=5.66679,
                     url=""),
            Hospital(name="Government Hospital Umunede", city="Umunede-Ogan Rd, Umunede", lat=6.24562, lon=6.30772,
                     url=""),
            Hospital(name="Owa-Oyibu Primary Health Centre", city="Idulaka Rd, Owa-Oyibu", lat=6.18137, lon=6.19174,
                     url=""),
            Hospital(name="Owa-Alero Government Hospital", city="Owa-Eke Rd, Owa-Aliosimi", lat=6.21982, lon=6.21651,
                     url=""),
            Hospital(name="Abavo Primary Health Centre", city="Ekuma, Abavo", lat=6.13373, lon=6.15210, url=""),
            Hospital(name="Agbor-Alidinma General Hospital", city="Ewuru Road, Agbor", lat=6.26440, lon=6.14770,
                     url=""),
            Hospital(name="Government Hospital Ofagbe", city="Ofagbe", lat=5.53712, lon=6.35289, url=""),
            Hospital(name="Delta State Government Hospital, Owhelogbo", city="Old Emevor Rd, Owhelogbo ", lat=5.59201,
                     lon=6.19493, url=""),
            Hospital(name="Ozoro Primary Health Centre", city="Oruamudhu St, Ozoro", lat=5.54870, lon=6.23822, url=""),
            Hospital(name="Ozoro General Hospital", city="Oruamudhu St, Ozoro", lat=5.54811, lon=6.23670, url=""),
            Hospital(name="Government Hospital Aviara", city="Aviara", lat=5.39010, lon=6.26589, url=""),
            Hospital(name="Warri South Local Government Cottage Hospital", city="Upper Erejuwa Rd, Warri", lat=5.52205,
                     lon=5.75031, url=""),
            Hospital(name="Uzere Primary Health Centre", city="Uzere", lat=5.33328, lon=6.23905, url=""),
            Hospital(name="Erhoke COttage Hospital", city="College Road, Kokori", lat=5.64176, lon=6.06593, url=""),
            Hospital(name="Government Hospital Olomoro", city="Oruabe", lat=5.41887, lon=6.12950, url=""),
            Hospital(name="Ashaka Government Hospital", city="Ushie Rd, Ashaka", lat=5.63243, lon=6.39060, url=""),
            Hospital(name="Ashaka General Hospital", city="Ushie Rd, Ashake", lat=5.63211, lon=6.39035, url=""),
            Hospital(name="General Hospital Aboh", city="Mbutu", lat=5.42157, lon=7.24102, url=""),
            Hospital(name="Umuolu General Hospital", city="Umuolu", lat=5.43130, lon=6.48587, url=""),
            Hospital(name="Kwale General Hospital", city="Kwale", lat=5.70558, lon=6.44199, url=""),
            Hospital(name="General Hospital Orerokpe", city="Orerokpe", lat=5.63270, lon=5.89132, url=""),
            Hospital(name="Government Hospital Mereje.", city="Mereje", lat=5.66380, lon=5.72023, url=""),
            Hospital(name="Mereje Primary Health Centre", city="Mereje", lat=5.66728, lon=5.71292, url=""),
            Hospital(name="Government Hospital Ibusa", city="Igbuzor, Umejei Rd, Ibusa", lat=5.03185, lon=5.66679,
                     url=""),
            Hospital(name="Ebu Primary Health Centre", city="Ebu", lat=6.48223, lon=6.60817, url=""),
            Hospital(name="Ebu General Hospital", city=" Illah-Okwe Rd, Ebu", lat=6.46942, lon=6.62947, url=""),
            Hospital(name="Akwukwu-Igbo General Hospital", city="Illah-Atumo Rd, Akwukwu Igbo", lat=6.35886,
                     lon=6.59324, url=""),
            Hospital(name="General Hospital Okwe", city="Okwe General Hospital Rd, Asaba", lat=6.16947, lon=6.74174,
                     url=""),
            Hospital(name="Patani General Hospital", city="Patani", lat=5.23569, lon=6.19255, url=""),
            Hospital(name="Patani Primary Health Centre", city="Patani", lat=5.22756, lon=6.19153, url=""),
            Hospital(name="Otor Udu General Hospital", city="Oleh-Okpari-Warri Rd, Udu lGC, Otudu", lat=5.45906,
                     lon=5.86813, url=""),
            Hospital(name="General Hospital Otu Jeremi", city=" Edjovhe-Otujerimi Rd, Otu Jeremi", lat=5.43296,
                     lon=5.87538, url=""),
            Hospital(name="Ewu General Hospital", city="Edjekota-Ewu Rd, Ewu", lat=5.38166, lon=5.98982, url=""),
            Hospital(name="Ewu Primary Health Centre", city="Edjekota-Ewu Rd, Oviri Olomu-Ogor Rd, Ewu", lat=5.38882,
                     lon=5.99336, url=""),
            Hospital(name="Umutu Primary Health Centre", city="Abraka - Umutu Rd, Agbor - Eku Rd, Umutu", lat=5.90853,
                     lon=6.22560, url=""),
            Hospital(name="Umutu Health Centre", city="Old Agbor- Sapele Rd, Umutu", lat=5.91450, lon=6.22603, url=""),
            Hospital(name="Obiaruku General Hospital", city="opposite idise, Obiaruku", lat=5.84561, lon=6.14491,
                     url=""),
            Hospital(name="Ekpan General Hospital", city="Hospital Rd, Ekpan, Warri", lat=5.56266, lon=5.74862, url=""),
            Hospital(name="Koko General Hospital", city="Koko Camp", lat=5.99871, lon=5.44544, url=""),
            Hospital(name="Abigborodo Primary Health Centre", city="Abigborudu", lat=5.89397, lon=5.53655, url=""),
            Hospital(name="Omadino Cottage Hospital", city="Omadino Town, Warri", lat=5.62656, lon=5.65092, url=""),
            Hospital(name="Ogbe-Ijoh Primary Health Centre", city="Ogbe-Ijoh", lat=5.47970, lon=5.73611, url=""),

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
            "fever (body hot / body dey hot/high temperature)",
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
            "fever wey last many days(fever)",
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
    "fever", "body", "body hot", "body dey hot", "chills", "body dey shake", "sweating", "too much sweat",
    "headache", "head dey pain", "fatigue", "body weak", "no strength", "nausea", "body dey turn",
    "vomiting", "dey vomit", "muscle aches", "body pain", "stomach pain", "stomach cramp",
    "belly twist"  "belly dey pain",
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
        html, body { height: 100vh; width: 100vw; margin: 0; padding: 0; }
        :root { 
            --transition-speed: 0.3s; 
            --light-bg: #ffffff; 
            --accent-color: #008000; 
            --light-text: #343a40; 
            --muted-light-text:#6c757d; 
            --muted-dark-text: #b0b0b0; 
            --dark-bg: #36393f; 
            --dark-text: #e0e0e0; 
        }
        body { 
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif; 
            margin: 0 auto; 
            height: 100%; 
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center; 
            text-align: center; 
            background-color: var(--light-bg); 
            color: var(--light-text); 
            transition: background-color var(--transition-speed), color var(--transition-speed); 
        }

        /* New container for semantic grouping */
        .splash-content {
            padding: 20px;
        }

        h1 { 
            /* Slightly reduced max font size for stability */
            font-size: clamp(3rem, 15vw, 3.5rem); 
            margin-bottom: 0.5rem; 
            color: var(--light-text); 
            transition: color var(--transition-speed); 
            font-weight: 700;
        }
        h3 { 
            font-size: clamp(1.5rem, 8vw, 1.8rem); 
            margin-top: 0; 
            color: var(--muted-light-text); 
            transition: color var(--transition-speed); 
            font-weight: 500;
        }
        .icon { color: var(--accent-color);}

        /* Dark Mode */
        @media (prefers-color-scheme: dark) {
            body { background-color: var(--dark-bg); color: var(--dark-text); }
            h1 { color: var(--dark-text); }
            h3 { color: var(--muted-dark-text); }
            .icon{color: var(--dark-text); }
        }

        /* Mobile View Adjustments (can be simplified now with clamp()) */
        @media (max-width: 480px) {
            h1 { font-size: clamp(2.5rem, 12vw, 2.5rem); }
            h3{ font-size: 1.1rem; }
        }
    </style>
</head>
<body>
    <div class="splash-content">
        <h1><i class="fa-solid fa-brain fa-icon-large icon"></i> CoreEfx AI</h1>
        <h3>Check Symptoms, Stay Healthy</h3>
    </div>
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
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        html { box-sizing: border-box; }
        *, *:before, *:after { box-sizing: inherit; }
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
            background:#ffffff;
            margin: 0 auto;
            padding: 20px ;
            display: flex;
            justify-content: center;
            align-item: center
        }
        h1 { color: #343a40; margin-bottom: 20px; text-align: center; }
        .icon { color: #008000; }
        
        /* 💡 FIX APPLIED HERE: Standardize Size and Look */
        form {
            background: none;
            padding: 40px 10px;
            width: 100%;
            max-width: 450px;
            position: sticky;
            /* FORCE MINIMUM HEIGHT FOR CONSISTENCY */
           /* Set to match the taller Signup form */
            /* DISTRIBUTE CONTENT EVENLY */
            display: flex;
            flex-direction: column;
            justify-content: space-around;
        }
        
        /* ... (rest of the CSS styles are the same) ... */
        
        input { width: 100%; padding: 15px; margin: 8px 0; border: 1px solid #ccc; border-radius: 20px; font-size: 16px; }
        input[type="text"], input[type="username"], input[type="email"], input[type="password"] { 
            padding-left: 45px; 
            padding-right: 45px; /* <-- ADD THIS LINE for the eye icon */
        }
        .input-group { position: relative; width: 100%; }
        .input-group .icon { position: absolute; left: 15px; top: 50%; transform: translateY(-50%); color: #008000; }
        button {
            width: 100%;
            padding: 15px;
            margin-top: 20px;
            background: #008000;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s ease;
            font-weight: bold;
        }
        button:hover { background: #006400; }
        a { color: #008000; text-decoration: none; font-weight: bold; transition: color 0.3s ease; }
        a:hover { color: #006400; }
        p { text-align: center; font-size: 14px; }
        .toggle-password {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
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

        .toggle-password:hover { color: #28a428; background: none; border:none; }
        /* ✅ Dark Mode */
        @media (prefers-color-scheme: dark) {
            body { background: #36393f; color: #eee; }
            input { background: #2c2f33; color: #eee; border: 1px solid #444; }
            input[type="username"], input[type="password"] { 
                padding-left: 45px; 
                padding-right: 45px;/* <-- ADD THIS LINE for the eye icon */
            }
            input::placeholder { color: #aaa; }
            button { background: #38b45a; color: white; }
            a { color: #38b45a; }
            h1 { color:#e0e0e0;}
            .icon { color:#e0e0e0;}
            a:hover { color:#2e8b43;}
            button:hover { background:#2e8b43;}
            .toggle-password { color:#38b45a;}
            .toggle-password:hover{ color:#2e8b43;}
            .input-group .icon { color:#38b45a;}
        }
        /* ✅ Mobile view adjustments */
        @media (max-width: 1020px) {
            input, button { font-size: 15px; padding: 12px; }
            h1 { font-size: 36px; margin-bottom: 30px; }
            .toggle-password{ right: 10px; left: auto; }
        }
        @media (max-width: 480px) {
            input, button { font-size: 14px; padding: 12px; }
            h1 { font-size: 34px; }
        }
    </style>
</head>
<body>
<form method="POST">
    <h1><i class="fa-solid fa-brain fa-icon-large icon"></i> CoreEfx AI</h1>
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
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        html { box-sizing: border-box; }
        *, *:before, *:after { box-sizing: inherit; }
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
            background:#ffffff;
            margin: 0 auto;
            padding: 20px ;
            display: flex;
            justify-content: center;
            align-item: center
        }
        h1 { color: #343a40; margin-bottom: 20px; text-align: center; }
        .icon { color: #008000; }
        
        /* Standardize Look/Spacing with Flexbox */
        form {
            position: sticky;
            background: none;
            padding: 40px 10px;
            border-radius: 10px;
            width: 100%;
            max-width: 450px;
            /* DISTRIBUTE CONTENT EVENLY */
            display: flex;
            flex-direction: column;
            justify-content: space-around;
        }
        
        input { width: 100%; padding: 15px; margin: 8px 0; border: 1px solid #ccc; border-radius: 20px; font-size: 16px; }
        input[type="text"], input[type="username"], input[type="email"], input[type="password"] { 
            padding-left: 45px; 
            padding-right: 45px; /* <-- ADD THIS LINE for the eye icon */
        }
        .input-group { position: relative; width: 100%; }
        .input-group .icon { position: absolute; left: 15px; top: 50%; transform: translateY(-50%); color: #008000; }
        /* Ensure the text doesn't flow under the eye icon on the right */
        input[type="text"], input[type="username"], input[type="email"], input[type="password"] { 
            padding-left: 45px; 
            padding-right: 45px; /* <-- ADD THIS LINE for the eye icon */
        }}
        button {
            width: 100%;
            padding: 15px;
            margin-top: 20px;
            background: #008000;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s ease;
            font-weight: bold;
        }
        button:hover { background: #006400; }
        a { color: #008000; text-decoration: none; font-weight: bold; transition: color 0.3s ease; }
        a:hover { color: #006400; }
        p { text-align: center; font-size: 14px; }
        .toggle-password {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
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
        .toggle-password:hover { color: #28a428; background: none; border:none; }
        /* ✅ Dark Mode */
        @media (prefers-color-scheme: dark) {
            body { background: #36393f; color: #eee; }
            input { background: #2c2f33; color: #eee; border: 1px solid #444; }
            input::placeholder { color: #aaa; }
            button { background: #38b45a; color: white; }
            a { color: #38b45a; }
            h1 { color:#e0e0e0;}
            .icon { color:#e0e0e0;}
            a:hover { color:#2e8b43;}
            button:hover { background:#2e8b43;}
            .toggle-password { color:#38b45a;}
            .toggle-password:hover{ color:#2e8b43;}
            .input-group .icon { color:#38b45a;}
        }
        /* ✅ Mobile view adjustments */
        @media (max-width: 1020px) {
            input, button { font-size: 15px; padding: 12px; }
            h1 { font-size: 36px; margin-bottom: 30px; }
            .toggle-password{ right: 10px; left: auto; }
        }
        @media (max-width: 480px) {
            input, button { font-size: 14px; padding: 12px; }
            h1 { font-size: 34px; }
        }
    </style>
</head>
<body>
<form method="POST">
    <h1><i class="fa-solid fa-brain fa-icon-large icon"></i> CoreEfx AI</h1>
    
    <div class="input-group">
        <i class="fa fa-address-card icon"></i>
        <input type="text" name="name" placeholder="Full Name" required />
    </div>

    <div class="input-group">
        <i class="fa fa-user icon"></i>
        <input type="username" name="username" placeholder="Username" required />
    </div>

    <div class="input-group">
        <i class="fa fa-envelope icon"></i>
        <input type="email" name="email" placeholder="Email" required />
    </div>

    <div class="input-group">
        <i class="fa fa-lock icon"></i>
        <input type="password" id="password" name="password" placeholder="Password" required />
        <div type="button" class="toggle-password" onclick="togglePassword('password', 'eye-icon-pass')">
            <i class="fa-solid fa-eye" id="eye-icon-pass"></i>
        </div>
    </div>
    
    <button type="submit">Sign Up</button>
    
    <p>
        Already have an account? <a href="/login">Login</a>
    </p>
     <p> By clicking "Sign Up" button, i expressly agree to CoreEfx AI <a href="/terms_of_service">Terms of Service</a> and understand that my account information will be used according to CoreEfx AI  <a href="/privacy_policy">Privacy Policy</a></p>
</form>
<script>
    function togglePassword(id, iconId) {
        const password = document.getElementById(id);
        const icon = document.getElementById(iconId);
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
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
            margin: 0 auto;
            padding: 80px 20px;
            height: 100%;
            background-color: var(--background-white);
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

        h3, h4 {
             /* A more prominent font for headings */
            text-align: center;
            margin-top: 25px;
            font-weight: 700; /* Bolder headings */
            transition: color var(--transition-speed); /* Smooth transition */
        }


        h2 {

            font-size: 2em;
            margin-bottom: 10px;
            color: var(--text-dark);
            margin-left: 25px;
            transition: color var(--transition-speed); /* Smooth transition */
        }
        .icon {
            color: var(--accent-color);
        }
        body.dark-mode  .icon{
            color: var(--text-dark);
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
            align-item: centre;
            color: var(--text-dark);
            padding: 15px 18px;
            text-align: centre;
            font-weight: 500;
            word-wrap: break-word;
            font-size: 2em;
            transition: color var(--transition-speed); /* Smooth transition */
            margin-top: 40px;
        }
        .intro-text { text-align: center; color: var(--text-muted); margin-bottom: 10px; font-size: 1.1em;  transition: color var(--transition-speed); /* Smooth transition */ }

        /* Tab Navigation Styles */
        .tabs {
            display: flex;
            justify-content: space-around;   /* ensures perfect spacing */
            align-items: center;
            width: 100%;
            padding: 9px 0;
        }
        
        .tab-button {
            background: none;
            border: none;
            flex: 1;                         /* each button same width */
            text-align: center;
            padding: 5px 0;
            color: var(--text-dark);
        }
        
        .tab-button i {
            display: block;
            font-size: 20px;
            margin-bottom: 3px;
        }
        
        .tab-button.active{
            font-weight: 1000;
        }
        
        .tab-button.active i{
             color: var(--accent-color);
        }
        .tab-button i:hover{
            color: var(--hover-accent);
        }
        
        .tab-button span {
            font-size: 14px;
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
            background-color: var(--background-white);
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


        form {
            padding: 0;
            border-radius: 8px;
            box-shadow: none;
            background-color: transparent;
            margin-top: 120px; 
            top: 150px;
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
            background-color: var(--background-light);
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
            background:var(--background-light);
            
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
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
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
             color: var(--text-dark);
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


        .results-section, .history-section, .feedback-form-section { /* Renamed feedback-section to feedback-form-section for clarity */
            margin-top: 40px;
            padding-top: 20px;
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

       
        
        .chat-btn{
            background:  var(--accent-color);
            align-item: center !important;
            border-radius: 30px;
            padding: 15px; 
            width: 25%;
            margin-top: 25px;
            text-align: center;
            margin-left: auto;
            margin-right: auto;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color var(--transition-speed);
            box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.12);
        }
        .chat-btn, a{
            text-decoration: none;
            color: white;            
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
            width: 100%;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: break-word;
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
        
       label b {
        margin-top: -60px;
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
          transform: scale(1.05);
        }


        .nav-menu {
          flex-grow: 1;
          display: none;
          flex-direction: column;
          position: absolute;
          top: 30px;
          right: -18px;
          background: var(--background-light);
          border-radius: 10px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
          padding: 10px 0;
          width: 210px;
          z-index: 1010;
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

         /* Responsive adjustments */
        @media (max-width: 1020px) {
            
            .tabs {
                justify-content: space-between;   /* better spacing on wider phones */
                padding: 8px 15px;
            }
        
            .tab-button i {
                font-size: 22px;
            }
        
            .tab-button span {
                font-size: 15px;
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
            
            .input-feed-icon{
                bottom: 300px; /* keeps it above footer on mobile */
                padding: 10px 12px;
                border-radius: 20px;
            }

            .chat-btn{
                width: 40%;
            }

            .mic-btn,
            .circle-btn {
                font-size: 18px;
            }
            
            .mic-btn{
                display: none;
            }

            .intro-welcome{
                max-width: 70%;
                font-size: 0.95em;
                padding: 12px 14px;
            }
            .nav-menu a:hover, .nav-menu button:hover {
                background: none;
                
            }

             /* Make tabs full width on smaller screens */
            .tab-button {
                flex-grow: 1; /* Make buttons take equal width */
                padding: 9px 15px;
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
           
            .footer-nav {
                height: 100px; /* taller footer on mobile */
            }
            
            h2 {
                font-size: 1.3em;
                text-align: left;
                
            }
            
            
           
           .input-feed-icon {
                bottom: 120px;
                width: 94%;
                border-radius: 18px;
                padding: 8px 10px;
           }


            .mic-btn,
            .circle-btn {
                font-size: 17px;
            }

             .intro-welcome{
                max-width: 65%;
                font-size: 0.9em;
                padding: 10px 12px;
            }
            
            .chat-btn{
                width: 50%;
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
                padding: 8px 0;
            }
        
            .tab-button i {
                font-size: 18px;    /* smaller icon */
            }
        
            .tab-button span {
                font-size: 12px;    /* smaller text */
            }


            

        }
        /* Footer container */
        .footer-nav {
          position: fixed;
          bottom: 0;
          left: 0;
          right: 0;
          width: 100%;
          background: var(--background-white); /* no border here */
          border-top: 0.2px solid var(--shadow-light);
          z-index: 1010;
          height: auto;
         
         
        }

        /* The actual container inside */
        .footer-container {
           width: 100%; 
           
           

        }

        /* Make sure content isn’t hidden behind footer 
        main, .container {
          padding-bottom: 100px;
        }*/
        .rotation{
            transform: rotate(62deg);
            align-self: center;
            font-size: 19px; 
        } 
        
    </style>
</head>
<body>
    <div class="container">
        <!-- New app-header for mobile alignment -->
        <header class="app-header">
            <h2><i class="fa-solid fa-brain fa-icon-large icon"></i> CoreEfx AI</h2>
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
                <a href="/privacy_policy"><i class="fa fa-lock"></i> Privacy Policy</a>
                <a href="/terms_of_service"><i class="fa fa-file-alt"></i> Terms of Service</a>

                <button class="logout-btn" onclick="window.location='{{ url_for('logout') }}'">
                    <i class="fa fa-sign-out-alt"></i> 
                    <span>Logout</span>
                </button>
              </div>
            </nav>

        </header>

        <div id="home" class="tab-content active">

            <h3>Welcome {{ current_user.username }}</h3>
            <p class="intro-text"><b>As your AI Health Advisor, I provide initial symptom guidance & help you find nearby hospitals. Please remember, a medical professional diagnosis is essential.</b></p>
           <a href="{{ url_for('chat') }}"> <div class="chat-btn"><span>Let's Chat</span></div></a>
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
                                <button type="submit" class="circle-btn"><i class="fa-solid fa-paper-plane rotation"></i></button>
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
    



    <script>
       
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
        }


        // All DOM-related initializations go here
        document.addEventListener('DOMContentLoaded', () => {
            console.log("DOM content loaded for initial setup.");
        });

    </script>
</body>
</html>
"""


# MIN_HYBRID_THRESHOLD = 5 # more lenient

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
            return redirect(url_for("home"))
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


@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    # 1. Handle Chat Messages (POST)
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get("symptoms", "").strip()
        lat_str = data.get("lat")
        lon_str = data.get("lon")

        result = []
        hospitals_list = []
        audio_file = ""

        # --- FIX: Initialize variables used in both branches ---
        valid_conditions = []
        ranked_conditions = []
        # ----------------------------------------------------

        # --- Symptom Logic (Start) ---
        if not user_input or not has_medical_relevance(user_input):
            # ... (Logic to set result and audio_file for invalid input) ...
            msg = "I could not identify any symptoms. Please describe your health condition or Visit one of these hospitals close to you, if available. Thank you!!!..."
            result = [msg]
            audio_file = generate_audio(msg) or ""
        else:
            # --- START: ML & Keyword Matching Logic ---
            # NOTE: Your full ML logic to populate ranked_conditions and valid_conditions
            # MUST be present here. The snippet below assumes it is.
            ml_results = ml_predict_condition(user_input, top_n=5, threshold=0.0)
            keyword_matches = check_symptoms(user_input, min_score_threshold=1, top_n=5)

            hybrid_results = {}
            for cond_name, prob in ml_results:
                hybrid_results[cond_name] = {"ml_conf": prob, "spacy_score": 0}
            for cond_name, score in keyword_matches:
                if cond_name in hybrid_results:
                    hybrid_results[cond_name]["spacy_score"] = score
                else:
                    hybrid_results[cond_name] = {"ml_conf": 0, "spacy_score": score}

            ranked_conditions = []  # Re-assign here is okay
            for cond_name, vals in hybrid_results.items():
                ml_conf = vals["ml_conf"]
                spacy_score = vals["spacy_score"]
                hybrid_score = (ml_conf * 100) + (spacy_score * 10)
                ranked_conditions.append((cond_name, hybrid_score))

            ranked_conditions.sort(key=lambda x: x[1], reverse=True)

            # This line successfully populates the initialized variable:
            valid_conditions = [c for c, score in ranked_conditions if score >= 6][:5]
            # --- END: ML & Keyword Matching Logic ---

            if valid_conditions:
                # ... (Logic to set result and audio_file based on valid conditions) ...
                advice_texts = []
                for cond_name in valid_conditions:
                    cond_key = cond_name.strip().lower()
                    if cond_key in symptom_data:
                        advice = symptom_data[cond_key]["advice"]
                        result.append(advice)
                        advice_texts.append(advice)
                audio_file = generate_audio(" ".join(advice_texts)) or ""
            else:
                msg = "\nI could not identify your condition. Please consult a doctor."
                result = [msg]
                audio_file = generate_audio(msg) or ""

        # --- HOSPITAL LOGIC (Must be here and correctly indented) ---
        if lat_str and lon_str:
            try:
                user_lat = float(lat_str)
                user_lon = float(lon_str)

                if user_lat == 0.0 and user_lon == 0.0:
                    result.append("\nWarning: Location data appears to be unavailable or invalid (0, 0).")
                else:
                    # Search logic (using the list hospitals_list)
                    hospitals_list = find_nearby_hospitals(user_lat, user_lon, radius_km=15)
                    if not hospitals_list:
                        hospitals_list = find_nearby_hospitals(user_lat, user_lon, radius_km=50)
                        if hospitals_list:
                            result.append("\nNote: Hospitals found outside the immediate 15km radius (within 50km).")
                        else:
                            result.append("\nNo nearby hospitals found within a 50km radius.")
            except ValueError:
                result.append("\nWarning: Could not process location data due to a format error.")
        else:
            result.append("\nWarning: Location access was not granted or coordinates were missing.")

        # --- Save Report (Must be here) ---
        new_report = SymptomReport(
            user_id=current_user.id,
            input_text=user_input,
            location=f"{lat_str},{lon_str}" if lat_str and lon_str else "N/A",
            result=" ".join(result)
        )
        db.session.add(new_report)
        db.session.commit()

        # 4. FINAL JSON RETURN (Must be here and unconditional)
        return jsonify({
            "result": result,
            "audio_file": audio_file,
            "hospitals": hospitals_list
        })

    # 2. Render Page (GET)
    return render_template('chat.html', hospitals=[], result=[], user_message="", audio_file="")
    
@app.route("/home", methods=["GET", "POST"])
@login_required
def home():
    # Initialize variables that the template expects, set to default values
    # These were previously used for chat output, but are now reset for the dashboard view
    result = []
    audio_file = ""
    user_input = ""
    feedback_message = ""

    # 1. Fetch Data for Dashboard Components (History & Map)

    # Fetch history reports for the 'history' tab
    history_reports = SymptomReport.query.filter_by(user_id=current_user.id).order_by(
        SymptomReport.timestamp.desc()
    ).all()

    # Fetch all hospitals for the initial map display on the Home tab
    all_hospitals = Hospital.query.all()
    hospitals = [
        {"name": h.name, "city": h.city, "lat": h.lat, "lon": h.lon, "url": h.url}
        for h in all_hospitals
    ]

    # 2. Handle POST Request (Now only handles Feedback Submission)
    if request.method == "POST":
        # We only check for feedback here, as symptom submissions are handled by the
        # separate AJAX call to the /chat route.
        if "feedback_message" in request.form:
            feedback_content = request.form["feedback_message"]
            if feedback_content:
                new_feedback = Feedback(message=feedback_content)
                db.session.add(new_feedback)
                db.session.commit()
                feedback_message = "Thank you for your feedback!"
            else:
                feedback_message = "Feedback message cannot be empty."

    # 3. Render the main dashboard template (main_template)
    return render_template_string(main_template, result=result, hospitals=hospitals,
                                  history_reports=history_reports, feedback_message=feedback_message,
                                  audio_file=audio_file, current_user=current_user, user_message=user_input)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
