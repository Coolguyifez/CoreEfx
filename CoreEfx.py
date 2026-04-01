from flask import Flask, request, redirect, render_template_string, render_template, jsonify, url_for, flash, send_from_directory
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
from itsdangerous import URLSafeTimedSerializer as Serializer




# --- TOKEN HELPERS ---
def get_reset_token(user_id):
    s = Serializer(app.secret_key)
    return s.dumps({'user_id': user_id})

def verify_reset_token(token, expires_sec=1800):
    s = Serializer(app.secret_key)
    try:
        data = s.loads(token, max_age=expires_sec)
        user_id = data['user_id']
    except Exception:
        return None
    return User.query.get(user_id)


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
db_path = "postgresql://auto:c3LRIMEx9HRBiA6dVZb0zs493hnPWeKT@dpg-d6osu1d6ubrc73bpa0s0-a.oregon-postgres.render.com/core_db_ljk2"
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

app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)




@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# === User model ===
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)


class SymptomReport(db.Model):
    """
    Represents a record of a user's symptom submission, including their input,
    detected location, the advice given, and a timestamp.
    """
    id = db.Column(db.Integer, primary_key=True)  # Unique identifier for each report
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_text = db.Column(db.Text)  # The raw text input from the user (symptoms)
    location = db.Column(db.String(100))  # User's approximate location (latitude,longitude string)
    result = db.Column(db.Text)  # The health advice/diagnosis provided by the system (increased length)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Time when the report was created
    severity = db.Column(db.String(20))  # Low, Moderate, High
    recipient = db.Column(db.String(50))  # "Myself" or "Someone else"
    last_diagnosed = db.Column(db.String(100))  # e.g., "Never", "Months ago"
    notice = db.Column(db.String(100))  # e.g., "Recently", "Long ago"
    age = db.Column(db.String(20))
    gender = db.Column(db.String(20))

    # 🔑 Link report to user
    user = db.relationship('User', backref=db.backref('reports', lazy=True))


class VitalsLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # Numerical data for analysis
    temperature = db.Column(db.Float)
    heart_rate = db.Column(db.Integer)
    bp_systolic = db.Column(db.Integer)
    bp_diastolic = db.Column(db.Integer)
    spo2 = db.Column(db.Integer)

    # Results
    severity = db.Column(db.String(20))  # Normal, Warning, Critical
    result = db.Column(db.Text)  # AI advice based on numbers
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)




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
    message = db.Column(db.String(1000))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)# The feedback message content
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
    severity_tag = db.Column(db.String(20), default="Moderate")  # Low, Moderate, High
    phone_number = db.Column(db.String(20), nullable=True)


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
                     url="https://lasuth.org.ng/", severity_tag="High", phone_number="8093699801" ),
            Hospital(name="University of Abuja Specialist Hospital", city="abuja", lat=8.965031007601917,
                     lon=7.064360769741204,
                     url="https://www.google.com/search?q=university+of+abuja+specialist+hospital+gwagwalada&sca", severity_tag="High", phone_number="08157699801"),

            Hospital(name="University of Benin Teaching Hospital (UBTH)",
                     city=" Benin Lagos Express Road, Uselu, Benin City", lat=6.3903335466504725, lon=5.611826787114936,
                     url="https://www.google.com/search?q=ubth&sca", severity_tag="High", phone_number="07075699801"),

                #---Delta state Governmental hosital-----
            Hospital(name="Delta State University Teaching Hospital (DELSUTH)", city="Oghara", lat=5.9405, lon=5.6698,
                     url="https://delsuth.com.ng/", severity_tag="High", phone_number="08031234501"),
            Hospital(name="Federal Medical Centre (FMC) Asaba", city="Asaba", lat=6.2121, lon=6.7122,
                     url="https://fmcasaba.gov.ng/", severity_tag="High", phone_number="08061234502"),
            Hospital(name="Asaba Specialist Hospital", city="Asaba", lat=6.2349, lon=6.6859,
                     url="https://asabaspecialisthospital.org/", severity_tag="High", phone_number="07031234503"),
            Hospital(name="Central Hospital Warri (Emergency Unit)", city="Warri", lat=5.5170, lon=5.7344,
                     url="https://deltastate.gov.ng/", severity_tag="High", phone_number="08131234504"),
            Hospital(name="Central Hospital Agbor (Emergency Unit)", city="Agbor", lat=6.2557, lon=6.1820,
                     url="https://deltastate.gov.ng/", severity_tag="High", phone_number="08021234505"),
            Hospital(name="Central Hospital Ughelli (Emergency Unit)", city="Ughelli", lat=5.4942, lon=5.9952,
                     url="https://deltastate.gov.ng/", severity_tag="High", phone_number="09011234506"),
            Hospital(name="Central Hospital Sapele", city="Sapele", lat=5.9004, lon=5.6811,
                     url="https://deltastate.gov.ng/", severity_tag="High", phone_number="07051234507"),
            Hospital(name="General Hospital Ogwashi-Uku", city="Ogwashi-Uku", lat=6.1816, lon=6.5302,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Umunede", city="Umunede", lat=6.2456, lon=6.3077,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Bomadi", city="Bomadi", lat=5.1684, lon=5.9139,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ibusa", city="Ibusa", lat=6.1770, lon=6.6340,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Kwale", city="Kwale", lat=5.7055, lon=6.4419,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Okwe", city="Asaba", lat=6.1694, lon=6.7417,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Orerokpe", city="Orerokpe", lat=5.6327, lon=5.8913,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Patani", city="Patani", lat=5.2356, lon=6.1925,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Otu-Jeremi", city="Ughelli South", lat=5.4329, lon=5.8753,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ewu", city="Ewu", lat=5.3816, lon=5.9898, url="https://deltastate.gov.ng/",
                     severity_tag="Moderate"),
            Hospital(name="General Hospital Obiaruku", city="Obiaruku", lat=5.8456, lon=6.1449,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ekpan", city="Ekpan", lat=5.5626, lon=5.7486,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Koko", city="Koko", lat=5.9987, lon=5.4454,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ozoro", city="Ozoro", lat=5.5481, lon=6.2367,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Oghara", city="Oghara", lat=5.9384, lon=5.6795,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Abraka", city="Abraka", lat=5.7894, lon=6.1022,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Akwukwu-Igbo", city="Oshimili North", lat=6.3588, lon=6.5932,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Onicha-Olona", city="Aniocha North", lat=6.3667, lon=6.5666,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Isheagu", city="Aniocha South", lat=6.0392, lon=6.5467,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Aboh", city="Ndokwa East", lat=5.5421, lon=6.4210,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ashaka", city="Ndokwa East", lat=5.6321, lon=6.3903,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Burutu", city="Burutu", lat=5.3502, lon=5.5165,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Orogun", city="Ughelli North", lat=5.6368, lon=6.1528,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="Baptist Medical Centre (Public-Partnered)", city="Eku", lat=5.7517, lon=5.9954,
                     url="https://deltastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="Cottage Hospital Omadino", city="Warri South", lat=5.6265, lon=5.6509,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Erhoike", city="Ethiope East", lat=5.6444, lon=6.0359,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Ogulagha", city="Burutu", lat=5.3513, lon=5.3430,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Abigborodo", city="Warri North", lat=5.8939, lon=5.5365,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Kiagbodo", city="Burutu", lat=5.2432, lon=5.8360,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Issele-Uku Health Centre", city="Aniocha North", lat=6.3187, lon=6.4762,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Boji-Boji Agbor Health Centre", city="Ika South", lat=6.2633, lon=6.1847,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Owa-Oyibu Health Centre", city="Ika North East", lat=6.1813, lon=6.1917,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Abavo Health Centre", city="Ika South", lat=6.1337, lon=6.1521,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Ozoro Primary Health Centre", city="Isoko North", lat=5.5487, lon=6.2382,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Oleh Primary Health Centre", city="Isoko South", lat=5.4779, lon=6.2042,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Aviara Health Centre", city="Isoko South", lat=5.3901, lon=6.2658,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Uzere Health Centre", city="Isoko South", lat=5.3332, lon=6.2390,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Olomoro Health Centre", city="Isoko South", lat=5.4188, lon=6.1295,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Ubulu-Uku Health Centre", city="Aniocha South", lat=6.2334, lon=6.4498,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Ebu Primary Health Centre", city="Oshimili North", lat=6.4822, lon=6.6081,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Illah Health Centre", city="Oshimili North", lat=6.4250, lon=6.6450,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Okpanam Health Centre", city="Oshimili North", lat=6.2500, lon=6.6500,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Ibusa Comprehensive Health Centre", city="Oshimili North", lat=6.1750, lon=6.6350,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Mereje Health Centre", city="Okpe", lat=5.6672, lon=5.7129, url="https://deltastate.gov.ng/",
                     severity_tag="Low"),
            Hospital(name="Effurun Health Centre", city="Uvwie", lat=5.5550, lon=5.7850,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Umutu Health Centre", city="Ukwuani", lat=5.9145, lon=6.2260,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Akumazi Health Centre", city="Ika North East", lat=6.2650, lon=6.3350,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Otolokpo Health Centre", city="Ika North East", lat=6.3050, lon=6.3850,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Umunede Health Post", city="Ika North East", lat=6.2480, lon=6.3100,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Idumuesah Health Centre", city="Ika North East", lat=6.2050, lon=6.3550,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Ekuku-Agbor Health Centre", city="Ika South", lat=6.1550, lon=6.2550,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Agbor-Obi Health Centre", city="Ika South", lat=6.2450, lon=6.1950,
                     url="https://deltastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Aladja Health Centre", city="Udu", lat=5.4850, lon=5.7650, url="https://deltastate.gov.ng/",
                     severity_tag="Low"),

            # --- Bayelsa State Governmental Hospital ---
            Hospital(name="Federal Medical Centre (FMC) Yenagoa", city="Yenagoa", lat=4.9372, lon=6.2666,
                     url="https://fmcyenagoa.org.ng/", severity_tag="High", phone_number="09021234561"),
            Hospital(name="Niger Delta University Teaching Hospital (NDUTH)", city="Okolobiri", lat=4.9850, lon=6.3210,
                     url="https://nduth.org.ng/", severity_tag="High", phone_number="08037771122"),
            Hospital(name="Diete Koki Memorial Specialist Hospital", city="Opolo, Yenagoa", lat=4.9480, lon=6.3050,
                     url="https://bayelsastate.gov.ng/", severity_tag="High", phone_number="07055544433"),
            Hospital(name="Bayelsa State Medical University Teaching Hospital", city="Onopa, Yenagoa", lat=4.9250,
                     lon=6.2650, url="https://bsmu.edu.ng/", severity_tag="High", phone_number="08122233344"),
            Hospital(name="Bayelsa Diagnostic Centre", city="Yenagoa", lat=4.9350, lon=6.2850,
                     url="https://bayelsastate.gov.ng/", severity_tag="High", phone_number="08099988877"),
            Hospital(name="General Hospital Sagbama", city="Sagbama", lat=5.1230, lon=6.2140,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ogbia", city="Ogbia Town", lat=4.6520, lon=6.3215,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ekeremor", city="Ekeremor", lat=5.0450, lon=5.7560,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Amassoma", city="Southern Ijaw", lat=4.8150, lon=6.1050,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Nembe", city="Nembe Town", lat=4.5410, lon=6.3980,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Brass", city="Twon-Brass", lat=4.3210, lon=6.2340,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Kaiama", city="Kolokuma/Opokuma", lat=5.1150, lon=6.3450,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Oporoma", city="Southern Ijaw", lat=4.6950, lon=6.0650,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Kolo", city="Ogbia", lat=4.7820, lon=6.3550,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Agudama-Epie", city="Yenagoa", lat=4.9810, lon=6.3620,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="Cottage Hospital Otuoke", city="Otuoke", lat=4.7940, lon=6.3110,
                     url="https://bayelsastate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="Cottage Hospital Agbere", city="Sagbama", lat=5.1850, lon=6.1550,
                     url="https://bayelsastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Akassa", city="Brass", lat=4.3120, lon=6.0550,
                     url="https://bayelsastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Odi", city="Kolokuma/Opokuma", lat=5.1850, lon=6.3010,
                     url="https://bayelsastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Zarama", city="Yenagoa", lat=5.0650, lon=6.4210,
                     url="https://bayelsastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Angalabiri", city="Sagbama", lat=5.0250, lon=5.9550,
                     url="https://bayelsastate.gov.ng/", severity_tag="Low"),
            Hospital(name="Cottage Hospital Bassambiri", city="Nembe", lat=4.5350, lon=6.4110,
                     url="https://bayelsastate.gov.ng/", severity_tag="Low"),
            *[Hospital(name=f"PHC Yenagoa Ward {i}", city="Yenagoa", lat=4.93 + i * 0.005, lon=6.26 + i * 0.005,
                       url="https://bayelsastate.gov.ng/phc", severity_tag="Low") for i in range(1, 11)],
            *[Hospital(name=f"PHC Southern Ijaw Sector {i}", city="Southern Ijaw", lat=4.80 + i * 0.005,
                       lon=6.10 + i * 0.005, url="https://bayelsastate.gov.ng/phc", severity_tag="Low") for i in
              range(1, 11)],
            *[Hospital(name=f"PHC Ogbia Community {i}", city="Ogbia", lat=4.65 + i * 0.005, lon=6.32 + i * 0.005,
                       url="https://bayelsastate.gov.ng/phc", severity_tag="Low") for i in range(1, 11)],
            *[Hospital(name=f"PHC Sagbama Unit {i}", city="Sagbama", lat=5.12 + i * 0.005, lon=6.21 + i * 0.005,
                       url="https://bayelsastate.gov.ng/phc", severity_tag="Low") for i in range(1, 10)],
            *[Hospital(name=f"PHC Ekeremor Ward {i}", city="Ekeremor", lat=5.04 + i * 0.005, lon=5.75 + i * 0.005,
                       url="https://bayelsastate.gov.ng/phc", severity_tag="Low") for i in range(1, 10)],
            *[Hospital(name=f"PHC Nembe Rural {i}", city="Nembe", lat=4.54 + i * 0.005, lon=6.39 + i * 0.005,
                       url="https://bayelsastate.gov.ng/phc", severity_tag="Low") for i in range(1, 9)],

            # --- Rivers state Governmental Body---
            Hospital(name="University of Port Harcourt Teaching Hospital (UPTH)", city="Choba, Port Harcourt",
                     lat=4.9005, lon=6.9287, url="https://upth.ng/", severity_tag="High", phone_number="08035550101"),
            Hospital(name="Rivers State University Teaching Hospital (RSUTH)", city="Old GRA, Port Harcourt",
                     lat=4.7799, lon=7.0142, url="https://rsuth.ng/", severity_tag="High", phone_number="08035550102"),
            Hospital(name="Braithewaite Memorial Specialist Hospital (BMSH)", city="Port Harcourt", lat=4.7785,
                     lon=7.0150, url="https://riversstate.gov.ng/", severity_tag="High", phone_number="08035550103"),
            Hospital(name="Professor Kelsey Harrison Hospital", city="Emenike St, Mile 1, PH", lat=4.7909, lon=6.9943,
                     url="https://riversstate.gov.ng/", severity_tag="High", phone_number="08035550104"),
            Hospital(name="Rivers State Dental and Maxillofacial Hospital", city="Garrison, Port Harcourt", lat=4.8055,
                     lon=7.0092, url="https://riversstate.gov.ng/", severity_tag="High", phone_number="08035550105"),
            Hospital(name="Obio Cottage Hospital", city="Rumuomasi, Port Harcourt", lat=4.8344, lon=7.0336,
                     url="https://obiocottage.com/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ahoada", city="Ahoada East", lat=5.0821, lon=6.6498,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Bori", city="Khana LGA", lat=4.6753, lon=7.3652,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Degema", city="Degema LGA", lat=4.7471, lon=6.7554,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Isiokpo", city="Ikwerre LGA", lat=4.9333, lon=6.8833,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Okrika", city="Okrika LGA", lat=4.7419, lon=7.0833,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Opobo", city="Opobo/Nkoro LGA", lat=4.5050, lon=7.5090,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Eleme", city="Eleme LGA", lat=4.7912, lon=7.1235,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Bonny", city="Bonny Island", lat=4.4450, lon=7.1700,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Terabor", city="Gokana LGA", lat=4.6610, lon=7.2510,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Omoku", city="Ogba/Egbema/Ndoni", lat=5.3410, lon=6.6550,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Eberi", city="Omuma LGA", lat=5.1200, lon=7.1100,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Okehi", city="Etche LGA", lat=5.1150, lon=7.0350,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Ngo", city="Andoni LGA", lat=4.4550, lon=7.2950,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="General Hospital Abua", city="Abua/Odual LGA", lat=4.9650, lon=6.6550,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="Cottage Hospital Azumini", city="Isiokpo", lat=4.9510, lon=6.9120,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="Cottage Hospital Obuama", city="Degema", lat=4.8120, lon=6.8210,
                     url="https://riversstate.gov.ng/", severity_tag="Moderate"),
            Hospital(name="Model PHC Rumuigbo", city="Obio/Akpor", lat=4.8450, lon=6.9850,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="Model PHC Iriebe", city="Obio/Akpor", lat=4.8610, lon=7.1020,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="Model PHC Ozuoba", city="Obio/Akpor", lat=4.8820, lon=6.9150,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="Model PHC Elekahia", city="Port Harcourt", lat=4.8120, lon=7.0250,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="Model PHC Rumuodomaya", city="Obio/Akpor", lat=4.8710, lon=7.0010,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="Model PHC Mile 3", city="Diobu, PH", lat=4.8010, lon=6.9850,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Amadi-Ama", city="Port Harcourt", lat=4.7750, lon=7.0450,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Eagle Island", city="Port Harcourt", lat=4.7850, lon=6.9750,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Bundu Ama", city="Port Harcourt", lat=4.7610, lon=7.0210,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Abuloma", city="Port Harcourt", lat=4.7950, lon=7.0550,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Onne", city="Eleme", lat=4.7210, lon=7.1550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Akpajo", city="Eleme", lat=4.8210, lon=7.1150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Ebubu", city="Eleme", lat=4.7850, lon=7.1450, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Choba", city="Obio/Akpor", lat=4.8950, lon=6.9050, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Aluu", city="Ikwerre", lat=4.9250, lon=6.9150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Rumuekini", city="Obio/Akpor", lat=4.8950, lon=6.9450,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Eneka", city="Obio/Akpor", lat=4.9120, lon=7.0350, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Atali", city="Obio/Akpor", lat=4.9050, lon=7.0650, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Eliozu", city="Obio/Akpor", lat=4.8750, lon=7.0250, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Rukpokwu", city="Obio/Akpor", lat=4.8910, lon=7.0010,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Igwuruta", city="Ikwerre", lat=4.9650, lon=7.0150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Omagwa", city="Ikwerre", lat=4.9950, lon=6.9250, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Ubima", city="Ikwerre", lat=5.1550, lon=6.9550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Elele", city="Ikwerre", lat=5.1150, lon=6.8250, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Omerelu", city="Ikwerre", lat=5.2550, lon=6.9150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Oyigbo", city="Oyigbo LGA", lat=4.8850, lon=7.1850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Afam", city="Oyigbo LGA", lat=4.8550, lon=7.2350, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Nonwa", city="Tai LGA", lat=4.7550, lon=7.2550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Botem", city="Tai LGA", lat=4.7150, lon=7.2850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Kpite", city="Tai LGA", lat=4.7350, lon=7.3050, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Mogho", city="Gokana LGA", lat=4.6550, lon=7.2850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Bodo", city="Gokana LGA", lat=4.6250, lon=7.2650, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC K-Dere", city="Gokana LGA", lat=4.6850, lon=7.2150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Baen", city="Khana LGA", lat=4.7150, lon=7.4150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Bane", city="Khana LGA", lat=4.6550, lon=7.4550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Taabaa", city="Khana LGA", lat=4.6250, lon=7.4250, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Kaani", city="Khana LGA", lat=4.7250, lon=7.3550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Ogu", city="Ogu/Bolo LGA", lat=4.7150, lon=7.2150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Bolo", city="Ogu/Bolo LGA", lat=4.6850, lon=7.1850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Wakama", city="Ogu/Bolo LGA", lat=4.7450, lon=7.1550,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Abonnema", city="Akuku-Toru", lat=4.7250, lon=6.7850,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Kula", city="Akuku-Toru", lat=4.3150, lon=6.6550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Bakana", city="Degema", lat=4.7450, lon=6.9550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Tombia", city="Degema", lat=4.7850, lon=6.8550, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Buguma", city="Asari-Toru", lat=4.7350, lon=6.8650, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Ido", city="Asari-Toru", lat=4.7150, lon=6.8450, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Abalama", city="Asari-Toru", lat=4.7550, lon=6.8850,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Obuama", city="Degema", lat=4.8050, lon=6.8150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Joinkrama", city="Ahoada West", lat=5.1550, lon=6.4850,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Akinima", city="Ahoada West", lat=5.0850, lon=6.4550,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Mbiama", city="Ahoada West", lat=5.0450, lon=6.4250,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Engenni", city="Ahoada West", lat=5.1250, lon=6.4150,
                     url="https://riversstate.gov.ng/phc", severity_tag="Low"),
            Hospital(name="PHC Okwuzi", city="ONELGA", lat=5.5150, lon=6.6850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Egbema", city="ONELGA", lat=5.4150, lon=6.7850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Kreigani", city="ONELGA", lat=5.2850, lon=6.5850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Eberi", city="Omuma", lat=5.1350, lon=7.1250, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Umuechem", city="Etche", lat=5.0150, lon=6.9850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Ulakwo", city="Etche", lat=5.0850, lon=7.1150, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low"),
            Hospital(name="PHC Ndashi", city="Etche", lat=5.1150, lon=7.1850, url="https://riversstate.gov.ng/phc",
                     severity_tag="Low")


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
            "muscle aches (body pain)",
            "feeling cold and hot",
            "body dey burn",
            "shivering",
            "cold dey catch me",
            "hotness for body",
            "fever and cold",
            "my body is vibrating",
            "strong body heat",
            "shaking fever",
            "high temperature",
            "drenching sweat",
            "malaise",
            "severe chills"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} signs of mild malaria. {subject} should rest and get a blood test soon."
                "\n\n🍎 **Diet Recommendation:** Eat light, energy-rich foods like pap, oats, or bananas. Drink plenty of water and coconut water to stay hydrated."
                "\n\n🛡️ **Precautions:** Sleep under a treated mosquito net and clear stagnant water around your home."
                "\n\n🚫 **Avoid:** Avoid heavy, oily, or spicy foods that can upset the stomach. Do not skip meals even if appetite is low."
            ),
            "moderate": (
                "\nYou likely {verb_has} malaria. Please visit the health center quickly for a blood test and ACT treatment."
                "\n\n🥗 **Diet Recommendation:** Focus on high-protein foods like boiled eggs or chicken soup to help the body recover. Fresh orange juice is great for Vitamin C."
                "\n\n🛡️ **Precautions:** Finish the full course of malaria drugs even if you start feeling better. Rest in a cool, ventilated room."
                "\n\n🚫 **Avoid:** Avoid alcohol, smoking, and strenuous physical exercise. Do not self-medicate without a confirmed test."
            ),
            "high": "\n🚨 {subject} {verb_is} showing severe malaria symptoms. Go to the hospital immediately for a drip or injections."
        }
    },

    "typhoid fever": {
        "symptoms": [
            "fever wey last many days(fever)",
            "stomach pain (belly dey pain)",
            "diarrhea (stool dey run)",
            "constipation (no fit stool)",
            "weakness (body weak well well)",
            "loss of appetite (no wan chop)",
            "tummy dey pain",
            "running stomach",
            "no solid stool",
            "belly full of wind",
            "belly dey grumble",
            "stomach dey pain me bad",
            "stomachache",
            "food no dey sweet me",
            "no fit chop anything",
            "abdominal discomfort",
            "poor appetite",
            "loss of taste",
            "prolonged fever",
            "bloating"
        ],
        "advice": {
            "low": (
                "\n{subject} should rest and drink only safe, boiled or bottled water. Monitor the fever closely."
                "\n\n🍎 **Diet Recommendation:** Stick to a semi-solid diet like soft rice (congee), custard, or well-cooked potatoes. These are easy on {possessive} stomach."
                "\n\n🛡️ **Precautions:** Always wash hands with soap after using the toilet and before touching food. Ensure all drinking water is boiled thoroughly."
                "\n\n🚫 **Avoid:** Avoid raw vegetables, unpeeled fruits, and salads that might have been washed in contaminated water. Do not drink tap water directly."
            ),
            "moderate": (
                "\nThis may be typhoid fever. {subject} {verb_needs} to visit the nearest health center for a Widal test and antibiotics."
                "\n\n🥗 **Diet Recommendation:** Increase fluid intake with coconut water, fresh fruit juices (no pulp), and electrolyte drinks to prevent dehydration from diarrhea."
                "\n\n🛡️ **Precautions:** {subject} should avoid preparing food for others until a doctor confirms the infection is gone. Sanitize all eating utensils."
                "\n\n🚫 **Avoid:** Avoid high-fiber foods like whole grains, nuts, and raw seeds, as they can irritate the intestines. Stay away from fried and spicy street foods."
            ),
            "high": "\n🚨 {subject} {verb_has} severe typhoid symptoms. Go to the hospital immediately to check for internal complications or the need for intravenous (IV) fluids."
        }
    },

    "common cold": {
        "symptoms": [
            "cough (dey cough)",
            "sore throat (throat dey pain)",
            "runny nose (catarrh dey come out)",
            "nasal congestion (nose block)",
            "mild fever (small body hot)",
            "catarrh",
            "throat dey scratch",
            "wetin I dey swallow dey pain",
            "I dey hawk phlegm",
            "my throat dey wound me",
            "nose dey leak water",
            "my head dey heavy with catarrh",
            "mucus discharge",
            "phlegm",
            "constant sneezing",
            "throat irritation",
            "stuffy nose"
        ],
        "advice": {
            "low": (
                "\nThis looks like a common cold. {subject} should rest and drink plenty of fluids."
                "\n\n🍎 **Diet Recommendation:** Drink warm lemon water with honey to soothe {possessive} throat. Eat warm pepper soup or chicken broth to help clear catarrh."
                "\n\n🛡️ **Precautions:** Use a clean tissue to blow {possessive} nose and dispose of it immediately. Wash hands frequently to avoid spreading the cold to others."
                "\n\n🚫 **Avoid:** Avoid very cold drinks or ice cream as they can irritate the throat. Stay away from dusty environments which can make the cough worse."
            ),
            "moderate": (
                "\n{subject} {verb_has} a persistent cold. If {possessive} throat pain gets worse, consider seeing a health worker."
                "\n\n🥗 **Diet Recommendation:** Increase intake of Vitamin C rich fruits like oranges, grapefruits, and tangerines. Garlic and ginger tea can help reduce congestion."
                "\n\n🛡️ **Precautions:** Use salt-water gargles twice a day to reduce throat swelling. Ensure {subject} {verb_is} sleeping in a well-ventilated but warm room."
                "\n\n🚫 **Avoid:** Avoid smoking or being near people who smoke, as the lungs are already sensitive. Avoid sharing towels or drinking glasses with family members."
            ),
            "high": "\n🚨 If {subject} {verb_is} having real trouble breathing or a very high fever with this cold, go to the clinic now. Severe congestion can sometimes lead to pneumonia."
        }
    },

    "influenza (flu)": {
        "symptoms": [
            "high fever (body dey hot well well)",
            "body aches (body pain)",
            "headache (head dey pain)",
            "cough (dey cough)",
            "fatigue (no strength)",
            "body dey pain everywhere",
            "big fever",
            "serious body weakness",
            "body weak die",
            "fever too high o",
            "ache all over",
            "I dey feel very weak",
            "severe body aches",
            "exhaustion",
            "general malaise",
            "widespread pain",
            "debilitating weakness"
        ],
        "advice": {
            "low": (
                "\nThis could be the flu. {subject} {verb_needs} plenty of bed rest and warm fluids."
                "\n\n🍎 **Diet Recommendation:** Eat light meals that are easy to digest, such as oats, pap, or soft-boiled yams. Stay hydrated with water and herbal teas."
                "\n\n🛡️ **Precautions:** Keep a thermometer handy to monitor the fever. {subject} should stay home from work or school until the fever has been gone for 24 hours."
                "\n\n🚫 **Avoid:** Avoid caffeinated drinks like coffee or strong soda, as they can lead to dehydration. Do not engage in heavy physical labor."
            ),
            "moderate": (
                "\n{subject} {verb_has} moderate flu symptoms. Monitor {possessive} breathing and visit a health worker if it does not improve in 3 days."
                "\n\n🥗 **Diet Recommendation:** Focus on protein-rich foods like beans or fish to help the body repair itself. Drink warm water with a pinch of salt and sugar (ORS) if {subject} {verb_is} sweating heavily."
                "\n\n🛡️ **Precautions:** Change bedsheets and pillowcases frequently to keep the environment clean. Use a humidifier or a bowl of hot water for steam inhalation."
                "\n\n🚫 **Avoid:** Avoid crowded places to prevent spreading the virus. Avoid self-medicating with strong antibiotics, as the flu is caused by a virus, not bacteria."
            ),
            "high": "\n🚨 {subject} {verb_has} severe flu signs. Go to the hospital immediately if {subject} {verb_is} gasping for air, chest pain is present, or if the fever refuses to come down with basic medicine."
        }
    },

    "diarrheal disease": {
        "symptoms": [
            "diarrhea (stool dey rush)",
            "vomiting (dey vomit)",
            "dehydration (mouth dry)",
            "stomach cramps (belly twist)",
            "stomachache",
            "stooling too much",
            "dey purge",
            "dey pour water",
            "I dey run to toilet always",
            "stomach dey push me",
            "dey vomit food",
            "dey purge steady",
            "frequent watery stools",
            "stomach upset",
            "abdominal cramps",
            "loose motion",
            "stomach rumbling"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} mild diarrhea. Start drinking ORS (salt and sugar solution) immediately to stay hydrated."
                "\n\n🍎 **Diet Recommendation:** Eat the BRAT diet: Bananas, Rice, Applesauce, and Toast. These help firm up the stool. Drink coconut water or salted rice water."
                "\n\n🛡️ **Precautions:** Wash hands thoroughly with soap after every toilet visit. Ensure all cooking utensils and plates are washed with clean, treated water."
                "\n\n🚫 **Avoid:** Avoid milk and dairy products, as they can make diarrhea worse. Stay away from greasy, fried foods and very sugary drinks or sodas."
            ),
            "moderate": (
                "\nThis may be a diarrheal infection. {subject} {verb_needs} to see a health worker if {subject} cannot stop vomiting."
                "\n\n🥗 **Diet Recommendation:** Sip on clear vegetable broths and diluted fruit juices (non-acidic). Eat small, frequent meals of boiled potatoes or plain pasta."
                "\n\n🛡️ **Precautions:** If {subject} {verb_is} handling food for others, stop immediately until the stooling stops. Use a disinfectant to clean the toilet area."
                "\n\n🚫 **Avoid:** Avoid caffeine (coffee/strong tea) and alcohol, which cause the body to lose more water. Avoid spicy peppers that can irritate the intestines."
            ),
            "high": "\n🚨 {subject} {verb_is} losing too much water. Go to the hospital for a drip immediately. Severe dehydration can lead to kidney failure or shock."
        }
    },

    "urinary tract infection (UTI)": {
        "symptoms": [
            "pain when urinating (when you piss e dey pain)",
            "frequent urge to urinate (dey always wan piss)",
            "lower stomach pain (lower belly dey pain)",
            "hot piss",
            "piss dey sting",
            "small piss dey come",
            "pissing dey burn me",
            "dey piss small small",
            "pain inside my private area",
            "need to piss always",
            "burning during urination",
            "dysuria",
            "pelvic pain",
            "bladder irritation",
            "blood in urine"
        ],
        "advice": {
            "low": (
                "\nThis looks like a urinary tract infection. {subject} should drink plenty of clean water to wash {possessive} system and not hold pee."
                "\n\n🍎 **Diet Recommendation:** Drink unsweetened cranberry juice if available. Eat water-rich fruits like watermelon and cucumbers to encourage frequent urination."
                "\n\n🛡️ **Precautions:** Always wipe from front to back after using the toilet to prevent bacteria from entering the tract. Wear loose-fitting cotton underwear."
                "\n\n🚫 **Avoid:** Avoid sugary drinks and artificial sweeteners, which can feed the bacteria. Limit spicy foods that might irritate the bladder."
            ),
            "moderate": (
                "\n{subject} {verb_needs} to visit a health center for a test and proper antibiotics for this infection."
                "\n\n🥗 **Diet Recommendation:** Incorporate probiotic-rich foods like plain yogurt (non-sugar) to help maintain healthy bacteria levels. Drink warm ginger tea for inflammation."
                "\n\n🛡️ **Precautions:** Empty the bladder immediately after sexual activity. Avoid using scented soaps or feminine sprays in the private area."
                "\n\n🚫 **Avoid:** Avoid alcohol and heavy caffeine, as these act as bladder irritants and can increase the feeling of urgency and pain."
            ),
            "high": "\n🚨 If {subject} {verb_has} high fever, chills, or back/side pain with these symptoms, seek medical care immediately. This suggests the infection may have reached the kidneys."
        }
    },

    "skin infection (rash/measles/chickenpox)": {
        "symptoms": [
            "rash (skin get small small spots)",
            "itchy skin (skin dey scratch)",
            "blisters (skin get water blister)",
            "red spots (red mark for body)",
            "fever (body hot)",
            "red spots on skin",
            "small small bumps",
            "skin burning",
            "smallpox dey my body",
            "my skin dey red",
            "I dey scratch every time",
            "skin dey vex me",
            "skin eruption",
            "severe itching",
            "vesicles",
            "hives",
            "dermatitis"
        ],
        "advice": {
            "low": (
                "\nThis may be a skin infection. {subject} should avoid scratching and keep the area clean and dry."
                "\n\n🍎 **Diet Recommendation:** Eat foods high in Vitamin C (oranges, guavas) and Zinc (beans, nuts) to help the skin heal faster."
                "\n\n🛡️ **Precautions:** Use mild, fragrance-free soap. Wear loose, breathable cotton clothing to avoid further irritation. Wash bedsheets and towels in hot water."
                "\n\n🚫 **Avoid:** Avoid sharing towels, sponges, or clothes with others to prevent spreading. Do not apply strong local herbs or 'concoctions' that can burn the skin."
            ),
            "moderate": (
                "\n{subject} should see a doctor for a medicated cream or medicine, especially if the rash is spreading."
                "\n\n🥗 **Diet Recommendation:** Drink plenty of water to keep the skin hydrated from the inside. Flaxseeds or fatty fish (like Titus/Mackerel) can help reduce skin inflammation."
                "\n\n🛡️ **Precautions:** Keep fingernails short to prevent skin damage and secondary infections from scratching. Apply a cool, damp cloth to itchy areas for relief."
                "\n\n🚫 **Avoid:** Avoid hot baths or showers, which can make itching worse. Stay away from harsh detergents and scented body creams until the skin clears."
            ),
            "high": "\n🚨 If the skin is peeling, very painful, or {subject} {verb_has} a very high fever, visit the hospital now. This could be a severe allergic reaction or systemic infection."
        }
    },

    "dehydration/heat exhaustion": {
        "symptoms": [
            "excessive thirst (mouth dey dry)",
            "dry mouth (tongue dry)",
            "dizziness (head dey turn)",
            "weakness (no strength)",
            "no urine (no dey piss)",
            "my head dey turn",
            "feel faint",
            "very thirsty",
            "I feel like say I go fall down",
            "mouth dry finish",
            "I can't pee",
            "body dey dry up",
            "extreme thirst",
            "light-headedness",
            "dry tongue",
            "oliguria",
            "syncope"
        ],
        "advice": {
            "low": (
                "\n{subject} may be dehydrated. {subject} should go to a cool place and drink clean water or ORS."
                "\n\n🍎 **Diet Recommendation:** Eat watery fruits like watermelon, oranges, and pineapples. Coconut water is excellent for replacing lost electrolytes."
                "\n\n🛡️ **Precautions:** Drink water even when not thirsty, especially when working under the sun. Carry a water bottle whenever going out."
                "\n\n🚫 **Avoid:** Avoid staying in poorly ventilated or extremely hot areas for too long. Do not wait until the mouth is dry before drinking water."
            ),
            "moderate": (
                "\n{subject} {verb_is} showing signs of heat exhaustion. Rest with legs raised and keep drinking fluids."
                "\n\n🥗 **Diet Recommendation:** Drink Oral Rehydration Salts (ORS) or a homemade Salt-Sugar Solution (SSS). Light soups (pepper soup without too much spice) can help replace salts."
                "\n\n🛡️ **Precautions:** Use a fan or wet towels to cool the body down. Wear light-colored and loose clothing to allow sweat to evaporate."
                "\n\n🚫 **Avoid:** Avoid alcohol and heavy caffeine (coffee/strong tea), as these make the body lose more water. Avoid heavy, hot meals that increase internal body heat."
            ),
            "high": "\n🚨 {subject} {verb_is} severely dehydrated. If {subject} cannot drink, has stopped sweating, or is losing consciousness, go to the hospital for a drip immediately."
        }
    },

    "headache": {
        "symptoms": [
            "headache (head dey pain)",
            "pressure in head",
            "migraine",
            "head heavy",
            "dey beat my head",
            "strong headache",
            "head dey burst",
            "my brain dey knock",
            "pain for head no gree stop",
            "strong pressure for inside head",
            "throbbing pain",
            "intense headache",
            "unrelenting pain",
            "cluster headache",
            "tension headache"
        ],
        "advice": {
            "low": (
                "\nThis may be due to stress or fatigue. {subject} should rest in a dark, quiet room and stay hydrated."
                "\n\n🍎 **Diet Recommendation:** Eat small, frequent meals to keep blood sugar steady. Magnesium-rich foods like bananas or leafy greens (Ugu) can help relax blood vessels."
                "\n\n🛡️ **Precautions:** Practice deep breathing or light neck stretches if the headache is due to tension. Limit screen time on phones or computers."
                "\n\n🚫 **Avoid:** Avoid loud noises and very bright lights. Do not skip meals, as hunger is a common trigger for headaches."
            ),
            "moderate": (
                "\n{subject} can take paracetamol. If the pain continues, {subject} should be checked for malaria or high blood pressure."
                "\n\n🥗 **Diet Recommendation:** Ginger tea can help if the headache is accompanied by light nausea. Stay away from very salty foods which can increase blood pressure."
                "\n\n🛡️ **Precautions:** Keep a 'headache diary' to see if certain foods or smells trigger the pain. Ensure {subject} is getting at least 7-8 hours of sleep."
                "\n\n🚫 **Avoid:** Avoid processed meats and foods with high MSG (certain bouillon cubes/seasonings). Limit caffeine intake, as 'caffeine withdrawal' can also cause headaches."
            ),
            "high": "\n🚨 If this is the worst headache {subject} {verb_has} ever felt, or if it comes with a stiff neck, confusion, or loss of vision, go to the hospital immediately."
        }
    },

    "body pain": {
        "symptoms": [
            "muscle pain",
            "joint pain",
            "back pain",
            "everywhere dey pain",
            "bone pain",
            "pain for back",
            "my bone dey pain",
            "pain for my waist",
            "my hand dey weak",
            "body dey pain me well well",
            "general weakness",
            "aching limbs",
            "joint stiffness",
            "musculoskeletal pain",
            "soreness"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} general body aches. This may be due to fatigue or minor stress. {subject} should rest and try a warm bath."
                "\n\n🍎 **Diet Recommendation:** Eat anti-inflammatory foods like fatty fish (Mackerel/Titus), ginger, and turmeric. Ensure {subject} {verb_is} getting enough Magnesium from bananas or green leafy vegetables (Ugu)."
                "\n\n🛡️ **Precautions:** Use a firm mattress for back pain. Practice light stretching if the pain is due to sitting for too long. Ensure proper posture when lifting heavy objects."
                "\n\n🚫 **Avoid:** Avoid heavy physical labor or intense exercise until the pain subsides. Do not stay in one position (sitting or standing) for too long."
            ),
            "moderate": (
                "\n{subject} {verb_has} significant body pain. {subject} can take paracetamol and stay hydrated, but visit a clinic if it persists."
                "\n\n🥗 **Diet Recommendation:** Drink plenty of water to flush out toxins. Bone broth or light pepper soup with ginger can help soothe aching muscles and joints."
                "\n\n🛡️ **Precautions:** Apply a warm compress to the aching area for 15 minutes at a time. Ensure {subject} gets 8 hours of sleep to allow the body to repair itself."
                "\n\n🚫 **Avoid:** Avoid self-medicating with strong 'Abo' (local herbal mixtures) which may affect the kidneys. Limit sugary snacks, as sugar can increase inflammation in the body."
            ),
            "high": "\n🚨 {subject} {verb_is} in severe pain. This could be a sign of a serious infection like malaria or meningitis. Please see a doctor immediately."
        }
    },

    "early pregnancy": {
        "symptoms": [
            "missed period (menstruation no come)",
            "nausea (body dey turn)",
            "vomiting (dey vomit for morning)",
            "breast tenderness (breast dey pain)",
            "fatigue (no strength)",
            "e don pass my time",
            "morning sickness",
            "cravings",
            "period no come this month",
            "I dey throw up in the morning",
            "I feel tired quickly",
            "breast dey heavy",
            "amenorrhea",
            "morning sickness",
            "food aversions",
            "tender breasts",
            "gestation"
        ],
        "advice": {
            "low": (
                "\nThese could be early pregnancy signs. {subject} should take a test to confirm the condition."
                "\n\n🍎 **Diet Recommendation:** Eat small, frequent meals rather than three large ones to manage nausea. Ginger biscuits or lemon water can help with morning sickness."
                "\n\n🛡️ **Precautions:** Start tracking the date of the last menstrual period. Rest as much as possible, as the body is using a lot of energy."
                "\n\n🚫 **Avoid:** Avoid all alcohol, tobacco, and unnecessary medications. Stop eating raw or undercooked eggs and meat."
            ),
            "moderate": (
                "\nIf confirmed, {subject} should visit a health center to start antenatal care and take pregnancy vitamins."
                "\n\n🥗 **Diet Recommendation:** Focus on Folic Acid and Iron-rich foods like beans, spinach, eggs, and fortified cereals. Drink plenty of clean water."
                "\n\n🛡️ **Precautions:** Wear a supportive bra if breasts are tender. Visit a dentist, as pregnancy can sometimes affect gum health."
                "\n\n🚫 **Avoid:** Avoid heavy lifting and exposure to harsh chemicals or fumes. Limit caffeine intake (coffee/strong tea/colas)."
            ),
            "high": "\n🚨 If {subject} {verb_has} severe lower belly pain, heavy bleeding, or constant fainting, go to the hospital immediately. This could be an ectopic pregnancy or other emergency."
        }
    },

    "Apollo/conjunctivitis": {
        "symptoms": [
            "red eye (eye red)",
            "itchy eye (eye dey scratch)",
            "watery eye (eye dey bring water)",
            "eye pain (eye dey pain)",
            "sticky discharge (eye gum when you wake)",
            "eye dey tear up",
            "eye full of pus",
            "eye dey burn",
            "my eye dey bring sand",
            "eye red like fire",
            "eye dey sticky",
            "I feel sand sand inside my eye",
            "purulent discharge",
            "eye mattering",
            "conjunctivitis",
            "gritty feeling",
            "burning eyes"
        ],
        "advice": {
            "low": (
                "\nThis may be Apollo. {subject} should avoid touching {possessive} eyes and wash {possessive} hands frequently."
                "\n\n🍎 **Diet Recommendation:** Eat foods rich in Vitamin A (carrots, sweet potatoes, palm oil) to support general eye health."
                "\n\n🛡️ **Precautions:** Use a clean, separate towel for the face. Change pillowcases daily while the infection is active to prevent re-infection."
                "\n\n🚫 **Avoid:** Avoid rubbing the eyes, as this spreads the infection and causes more damage. Do not share towels, sponges, or eye makeup with anyone."
            ),
            "moderate": (
                "\n{subject} {verb_needs} to visit a health center for proper antibiotic eye drops."
                "\n\n🥗 **Diet Recommendation:** Stay hydrated and eat leafy greens. Omega-3 fats found in nuts or fish can help with eye surface inflammation."
                "\n\n🛡️ **Precautions:** Wear sunglasses if light bothers the eyes. Apply a cool (not ice cold) compress using a clean cloth to the closed eyelids for comfort."
                "\n\n🚫 **Avoid:** Avoid using 'urine', 'sugar water', or 'lemon juice' in the eyes; these are dangerous and can cause blindness. Do not wear contact lenses until the infection is gone."
            ),
            "high": "\n🚨 If {subject} cannot see clearly, has a very high fever, or has intense pain that feels like it's behind the eye, {subject} must see an eye doctor immediately."
        }
    },

    "blurred vision": {
        "symptoms": [
            "blurred vision (no see clear)",
            "eye pain (eye dey pain)",
            "light sensitivity (eye no like light)",
            "vision loss (eye dey blind small small)",
            "eye dey confuse me",
            "heavy head",
            "BP high",
            "I dey see double",
            "eye dey tear with light",
            "my sight don weak",
            "can't see far",
            "visual impairment",
            "seeing halos",
            "double vision",
            "photophobia",
            "loss of sight"
        ],
        "advice": {
            "low": (
                "\n{possessive} vision might be slightly blurry. {subject} should rest {possessive} eyes and avoid bright screens for now."
                "\n\n🍎 **Diet Recommendation:** Consume more dark leafy greens (Ugu/Spinach) and eggs, which contain Lutein to protect the eyes."
                "\n\n🛡️ **Precautions:** Ensure there is proper lighting when reading or working. Follow the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds."
                "\n\n🚫 **Avoid:** Avoid staring at phone or TV screens in a dark room. Do not use over-the-counter eye drops without knowing the cause of the blurriness."
            ),
            "moderate": (
                "\n{subject} {verb_has} noticeable eye trouble. {subject} {verb_needs} an urgent check-up with an eye specialist (ophthalmologist)."
                "\n\n🥗 **Diet Recommendation:** Control blood sugar and blood pressure through diet (low salt, low sugar), as these are major causes of vision problems."
                "\n\n🛡️ **Precautions:** If {subject} has a history of high blood pressure, check it immediately. Wear protective eyewear if working in dusty or bright environments."
                "\n\n🚫 **Avoid:** Avoid driving or operating machinery if vision is not clear. Do not buy 'reading glasses' from the roadside without a proper eye test."
            ),
            "high": "\n🚨 {subject} {verb_is} experiencing rapid vision loss, 'flashing lights', or intense eye pain. This is a medical emergency (possible Glaucoma or Retinal Detachment). Go to an eye clinic right now."
        }
    },

    "pneumonia": {
        "symptoms": [
            "fast breathing (dey breathe fast)",
            "difficulty breathing (no fit breathe well)",
            "cough with mucus (cough dey bring phlegm)",
            "high fever (body hot well well)",
            "chest pain (chest dey pain)",
            "short breath",
            "hard to breathe",
            "pain for chest",
            "I dey struggle to breathe",
            "chest dey make noise",
            "coughing out yellow tin",
            "breath no dey enter well",
            "rales",
            "productive cough",
            "shortness of breath",
            "pleuritic chest pain",
            "laboured breathing"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} a chest infection. Monitor {possessive} breathing closely and stay warm."
                "\n\n🍎 **Diet Recommendation:** Eat warm, easy-to-digest meals like oats or pap. Garlic and onions have mild natural antimicrobial properties that can be added to soups."
                "\n\n🛡️ **Precautions:** Keep the chest warm. Use a humidifier or sit in a steamy bathroom to help loosen mucus in the lungs."
                "\n\n🚫 **Avoid:** Avoid cold drinks and sitting directly under a fan or air conditioner. Do not smoke or stay near people who are smoking."
            ),
            "moderate": (
                "\nThis could be pneumonia. {subject} {verb_needs} a medical exam and likely antibiotics from a health center."
                "\n\n🥗 **Diet Recommendation:** Drink warm water with honey and lemon to soothe the throat and chest. High-protein foods like chicken or beans help the body fight the infection."
                "\n\n🛡️ **Precautions:** Get plenty of bed rest. Sleep with an extra pillow to prop the head up, which can make breathing easier."
                "\n\n🚫 **Avoid:** Avoid heavy oily foods that can make {subject} feel more sluggish. Do not stop taking prescribed antibiotics early, even if {subject} feels better."
            ),
            "high": "\n🚨 If {subject} {verb_is} struggling to breathe, has blue lips, or is very confused, seek oxygen and emergency medical care immediately."
        }
    },

    "tuberculosis (tb)": {
        "symptoms": [
            "cough wey last more than 2 weeks",
            "coughing blood (cough dey bring blood)",
            "weight loss (body dey reduce)",
            "night sweats (body dey sweat for night)",
            "chest pain (chest dey pain)",
            "cough for long time",
            "spit blood",
            "perspire for night",
            "cough wey no end",
            "I cough blood",
            "I dey perspire for night",
            "chest pain when I cough",
            "chronic cough",
            "hemoptysis",
            "drenching night sweats",
            "cachexia",
            "wasting syndrome"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} a persistent cough. Because it has lasted long, {subject} should go for a free TB test at a health center."
                "\n\n🍎 **Diet Recommendation:** Focus on a high-calorie, high-protein diet (eggs, milk, beans, meat) to prevent further weight loss."
                "\n\n🛡️ **Precautions:** Cover the mouth with a tissue or elbow when coughing. Ensure the living area has plenty of fresh air and sunlight, which kills TB bacteria."
                "\n\n🚫 **Avoid:** Avoid crowded places where the infection could be spread to others. Do not smoke, as it further damages the lungs."
            ),
            "moderate": (
                "\nThis could be tuberculosis. It can spread to others in the house. {subject} must visit a health worker for testing and free treatment immediately."
                "\n\n🥗 **Diet Recommendation:** Take Vitamin B6 rich foods (bananas, potatoes) and a multivitamin, as TB treatment can sometimes cause vitamin deficiencies."
                "\n\n🛡️ **Precautions:** Ensure everyone living with {subject} is also screened. TB treatment is long (usually 6 months); {subject} must commit to taking it every single day."
                "\n\n🚫 **Avoid:** Avoid alcohol during treatment, as it can cause severe liver problems when mixed with TB drugs. Do not use local 'cough mixtures' to hide the symptoms."
            ),
            "high": "\n🚨 {subject} {verb_is} coughing blood and losing weight rapidly. {subject} {verb_needs} immediate hospital admission for specialized TB care and isolation."
        }
    },

    "diabetes": {
        "symptoms": [
            "frequent urination (dey piss too much)",
            "excessive thirst (always dey thirsty)",
            "always hungry (too much hunger)",
            "weight loss (body dey slim down)",
            "wounds no dey heal quick",
            "too much piss",
            "dry throat always",
            "sugar level high",
            "piss too much",
            "always thirsty",
            "always want to eat",
            "wound no dey close",
            "sugar dey high",
            "polyuria",
            "polydipsia",
            "polyphagia",
            "unhealed sores",
            "hyperglycemia"
        ],
        "advice": {
            "low": (
                "\nThis may be diabetes. {subject} should reduce sugar intake and go for a blood sugar test."
                "\n\n🍎 **Diet Recommendation:** Switch to whole grains like local brown rice or 'Ofada'. Eat plenty of fiber from vegetables (Ugu, Okra) to slow down sugar absorption."
                "\n\n🛡️ **Precautions:** Start a daily walking routine (30 minutes). Learn to check blood sugar at home if possible."
                "\n\n🚫 **Avoid:** Avoid sugary soft drinks, malt drinks, and white bread. Stop adding extra sugar to tea, pap, or garri."
            ),
            "moderate": (
                "\n{subject} {verb_needs} to see a doctor to manage {possessive} sugar levels and get proper medication."
                "\n\n🥗 **Diet Recommendation:** Focus on 'Low Glycemic Index' foods. Beans are excellent. Limit high-sugar fruits like very ripe pineapples or watermelons; choose garden eggs or avocados instead."
                "\n\n🛡️ **Precautions:** Check the feet daily for any small cuts or sores, as these can turn into serious wounds for diabetics. Wear comfortable, well-fitting shoes."
                "\n\n🚫 **Avoid:** Avoid skipping meals, as this can cause blood sugar to swing dangerously. Do not use 'herbal bitters' as a replacement for hospital medication."
            ),
            "high": "\n🚨 If {subject} {verb_is} confused, has 'fruity-smelling' breath, is vomiting, or is very weak, {possessive} sugar may be dangerously high. Go to the hospital now."
        }
    },

    "hypertension (high blood pressure)": {
        "symptoms": [
            "headache (head dey pain often)",
            "dizziness (head dey turn)",
            "blurred vision (eye no see clear)",
            "chest pain (chest dey pain)",
            "sometimes no symptoms at all",
            "BP high",
            "blood pressure is high"
            "heavy head",
            "eye dey confuse me",
            "my head dey turn me",
            "BP don high",
            "I dey see black spot",
            "occipital headache",
            "vertigo",
            "seeing spots",
            "epistaxis",
            "chest tightness"
        ],
        "advice": {
            "low": (
                "\n{possessive} blood pressure may be slightly high. {subject} should rest, reduce salt, and check the BP again later."
                "\n\n🍎 **Diet Recommendation:** Use natural spices like garlic, ginger, and onions instead of salt or bouillon cubes. Eat potassium-rich foods like bananas and potatoes to help lower pressure."
                "\n\n🛡️ **Precautions:** Sit quietly for 5 minutes before checking blood pressure. Avoid stressful arguments or heavy physical exertion when the head feels heavy."
                "\n\n🚫 **Avoid:** Avoid 'white' salt and salty snacks (chips, salted nuts). Limit alcohol and kola nuts, which can spike blood pressure."
            ),
            "moderate": (
                "\nThis looks like hypertension. {subject} {verb_needs} to see a doctor for a proper BP check and lifestyle advice."
                "\n\n🥗 **Diet Recommendation:** Follow the DASH diet—lots of vegetables (Ugu, garden eggs), fruits, and low-fat proteins. Drink plenty of water and limit red meat."
                "\n\n🛡️ **Precautions:** Start a daily walking routine (30 mins) if the doctor clears it. Take any prescribed BP medicine at the same time every day—do not skip doses even if feeling fine."
                "\n\n🚫 **Avoid:** Avoid processed meats (sausages, canned meats) which are very high in sodium. Stop smoking immediately, as it narrows blood vessels and increases stroke risk."
            ),
            "high": "\n🚨 {possessive} blood pressure is very high. This can lead to a stroke or heart failure. Go to the hospital immediately."
        }
    },

    "heart disease": {
        "symptoms": [
            "chest pain (chest dey pain or tight)",
            "shortness of breath (no fit breathe well)",
            "swelling in legs (leg dey swell)",
            "tired easily (body weak quick quick)",
            "fast or irregular heartbeat (heart dey beat fast)",
            "heart dey beat quick",
            "dey gasp for air",
            "leg dey swell up",
            "heart dey jump",
            "I dey tire quick",
            "pain for my chest like fire",
            "chest tightness",
            "palpitations",
            "edema",
            "arrhythmia",
            "ankle swelling"
        ],
        "advice": {
            "low": (
                "\n{subject} should avoid stress and salty foods. Monitor if the chest tightness continues."
                "\n\n🍎 **Diet Recommendation:** Eat heart-healthy fats found in avocados and olive oil (or local palm oil in very small amounts). Fiber from oats and beans helps clear 'bad' cholesterol."
                "\n\n🛡️ **Precautions:** Maintain a healthy weight to reduce the workload on the heart. Avoid sudden heavy lifting."
                "\n\n🚫 **Avoid:** Avoid fried foods and trans-fats (margarine, fast food). Limit high-sugar foods which contribute to heart inflammation."
            ),
            "moderate": (
                "\nThis may be a heart problem. {subject} {verb_needs} a check-up with a cardiologist for an ECG or scan."
                "\n\n🥗 **Diet Recommendation:** Focus on 'Titus' or Mackerel fish (Omega-3) which protects the heart. Reduce total carbohydrate intake (yam, fufu) if overweight."
                "\n\n🛡️ **Precautions:** If legs are swelling, try to keep them elevated when sitting. Monitor how many pillows are needed to breathe comfortably at night."
                "\n\n🚫 **Avoid:** Avoid high-caffeine energy drinks which can cause irregular heartbeats. Avoid 'bitters' or local herbs that claim to 'wash the heart'—they can be dangerous."
            ),
            "high": "\n🚨 {subject} {verb_is} showing signs of a heart attack or acute heart failure. Go to the emergency room right now."
        }
    },

    "malnutrition": {
        "symptoms": [
            "weight loss (body too slim)",
            "swollen feet (leg dey swell from kwashiorkor)",
            "thin arms (hand dey thin)",
            "child no grow well",
            "weakness (no strength)",
            "body thin",
            "belly big",
            "child dey small",
            "child dey thin",
            "head dey big for small body",
            "body dey waste",
            "no strength for child",
            "wasting",
            "stunted growth",
            "marasmus",
            "kwashiorkor",
            "underweight"
        ],
        "advice": {
            "low": (
                "\n{possessive} symptoms may be due to mild malnutrition. Please provide balanced meals with proteins (beans, eggs, fish) and vitamins."
                "\n\n🍎 **Diet Recommendation:** Combine grains and legumes (e.g., Rice and Beans) to make a complete protein. Add crayfish to pap or soups for extra nutrients."
                "\n\n🛡️ **Precautions:** Ensure the person is dewormed, as worms can steal nutrients from food. Wash hands before eating to prevent infections."
                "\n\n🚫 **Avoid:** Avoid 'empty' calories like white sugar and sodas. Do not rely only on starchy foods (yam/cassava) without adding protein."
            ),
            "moderate": (
                "\n{subject} {verb_has} signs of moderate malnutrition. {subject} {verb_needs} a nutrition plan and supplements from a health center."
                "\n\n🥗 **Diet Recommendation:** Use 'Ready-to-Use Therapeutic Food' (RUTF) if provided by a clinic. Feed small, frequent meals if the appetite is low."
                "\n\n🛡️ **Precautions:** Monitor weight weekly. Ensure the child/patient is immunized against diseases that can worsen malnutrition (like Measles)."
                "\n\n🚫 **Avoid:** Avoid giving only watery pap; thicken it with milk, groundnut paste, or soya bean powder. Do not use local 'purging' medicines which cause further weight loss."
            ),
            "high": "\n🚨 This is severe malnutrition. {subject} must be taken to a stabilization center or hospital immediately for therapeutic feeding to prevent organ failure."
        }
    },

    "stomach ulcer": {
        "symptoms": [
            "burning pain for stomach (pain for stomach like fire)",
            "pain after eating (pain after you chop)",
            "vomiting blood (dey vomit blood)",
            "black stool (poo black)",
            "feel full quick (belly full fast)",
            "tummy dey burn",
            "stomach dey worry me",
            "stomach dey bleed",
            "dark tarry stool",
            "nausea",
            "indigestion"
        ],
        "advice": {
            "low": (
                "\n{subject} should avoid spicy foods and soda. Do not take pain killers like Ibuprofen on an empty stomach."
                "\n\n🍎 **Diet Recommendation:** Eat bland foods like boiled potatoes, oats, or bananas. Cabbage juice has been known to help soothe the stomach lining."
                "\n\n🛡️ **Precautions:** Eat smaller meals more frequently rather than three large ones. Drink plenty of water between meals rather than during them."
                "\n\n🚫 **Avoid:** Avoid hot pepper, citrus fruits (lemons/oranges), and vinegar. Avoid smoking, which increases stomach acid."
            ),
            "moderate": (
                "\nThis could be an ulcer. {subject} {verb_needs} to see a doctor for treatment to protect the stomach lining."
                "\n\n🥗 **Diet Recommendation:** Yogurt with live cultures can help balance gut bacteria. Steamed fish or chicken is better than fried or highly seasoned meats."
                "\n\n🛡️ **Precautions:** Manage stress, as it can increase acid production. Finish the full course of ulcer medication (like Omeprazole) as directed by the doctor."
                "\n\n🚫 **Avoid:** Avoid NSAIDs (Ibuprofen, Aspirin, Diclofenac) as these can cause the ulcer to bleed. Avoid coffee and strong tea."
            ),
            "high": "\n🚨 {subject} {verb_is} showing signs of internal bleeding (black, tarry stool or vomiting blood). This is a life-threatening emergency. See a doctor immediately."
        }
    },

    "asthma": {
        "symptoms": [
            "shortness of breath (no fit breathe well)",
            "wheezing (chest dey make noise like whistle)",
            "tight chest (chest heavy)",
            "cough (dey cough mostly for night)",
            "hard to breathe (struggle to breathe)",
            "short breath",
            "chest dey hold me",
            "breathing dey sound",
            "cough for night",
            "gasping for air",
            "difficulty breathing",
            "chest tightness"
        ],
        "advice": {
            "low": (
                "\n{subject} should stay away from dust and smoke. Keep the inhaler close."
                "\n\n🍎 **Diet Recommendation:** Eat foods high in Vitamin D (eggs, fish) and Vitamin C (citrus, if it doesn't trigger you) to support lung health. Ginger can help reduce airway inflammation."
                "\n\n🛡️ **Precautions:** Use a damp cloth for dusting instead of a broom. Keep windows closed during high-pollen or high-soot seasons."
                "\n\n🚫 **Avoid:** Avoid strong perfumes, incense, and mosquito coils. Avoid cold drinks if they trigger a cough."
            ),
            "moderate": (
                "\n{possessive} asthma seems to be acting up. {subject} {verb_needs} to use a preventer inhaler and see a doctor."
                "\n\n🥗 **Diet Recommendation:** Omega-3 fatty acids from fish can help lower lung inflammation. Keep meals light at night to avoid acid reflux, which can trigger asthma."
                "\n\n🛡️ **Precautions:** Always carry a rescue inhaler (Salbutamol). Learn the 'Peak Flow' to monitor lung strength at home."
                "\n\n🚫 **Avoid:** Avoid pets with fur if they cause sneezing or wheezing. Do not stop taking 'preventer' inhalers even when feeling perfectly fine."
            ),
            "high": "\n🚨 {subject} cannot breathe well and the inhaler is not working. Go to the hospital immediately for oxygen and nebulization."
        }
    },
    "tonsillitis": {
        "symptoms": [
            "throat dey pain when I swallow",
            "small fever",
            "red for throat",
            "neck dey swell",
            "sore throat",
            "painful swallowing",
            "swollen tonsils",
            "fever",
            "lymph nodes swelling"
        ],
        "advice": {
            "low": (
                "\nThis looks like mild tonsillitis. {subject} should drink warm water and gargle with salt water to soothe the pain."
                "\n\n🍎 **Diet Recommendation:** Stick to soft foods like pap, mashed potatoes, or yogurt. Warm honey and lemon water can coat the throat and reduce pain."
                "\n\n🛡️ **Precautions:** Get plenty of rest to allow the immune system to fight the infection. Replace the toothbrush after the infection clears to avoid re-infection."
                "\n\n🚫 **Avoid:** Avoid very crunchy or hard foods (like fried plantain chips) that can scratch the throat. Avoid sharing drinking cups or cutlery."
            ),
            "moderate": (
                "\n{subject} {verb_has} moderate tonsillitis. {subject} likely {verb_needs} antibiotics. Please visit a health worker if the fever stays high."
                "\n\n🥗 **Diet Recommendation:** Cold foods like ice cream or cold yogurt can sometimes numb the throat and provide relief. Warm chicken broth is excellent for hydration."
                "\n\n🛡️ **Precautions:** If antibiotics are prescribed, {subject} MUST finish the whole pack even if the pain stops. This prevents heart and kidney complications (Rheumatic fever)."
                "\n\n🚫 **Avoid:** Avoid smoking or being around smoke, which irritates the tonsils. Do not shout or strain the voice."
            ),
            "high": "\n🚨 If {subject} {verb_is} drooling, cannot swallow saliva, or has a muffled 'hot potato' voice, go to the emergency room immediately. This could be a peritonsillar abscess."
        }
    },

    "hepatitis (viral)": {
        "symptoms": [
            "yellow eye and skin",
            "dark piss",
            "body weak",
            "stomach pain for right side",
            "fever",
            "jaundice",
            "yellow urine",
            "fatigue",
            "abdominal pain",
            "nausea",
            "loss of appetite"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} signs of liver irritation. Rest well and avoid all alcohol and herbal mixtures."
                "\n\n🍎 **Diet Recommendation:** Focus on easy-to-digest carbohydrates like rice and oats. Eat small, frequent meals if nausea is present. Leafy greens help provide vital vitamins."
                "\n\n🛡️ **Precautions:** Practice good hygiene. If it is Hepatitis A, it spreads through contaminated food/water. Do not share needles or razors."
                "\n\n🚫 **Avoid:** Avoid fatty, oily, and fried foods which put a strain on the liver. Avoid processed sugars."
            ),
            "moderate": (
                "\nThis may be Hepatitis. {subject} {verb_needs} a blood test (HBsAg) at the hospital to know the type."
                "\n\n🥗 **Diet Recommendation:** Increase protein intake from lean sources like beans or fish (if the liver can handle it) to help repair liver cells. Avoid heavy red meat."
                "\n\n🛡️ **Precautions:** Rest is the most important treatment for the liver. Inform close family members so they can get tested and vaccinated."
                "\n\n🚫 **Avoid:** **Stricly avoid alcohol.** Avoid self-medicating with Paracetamol (which is processed by the liver) unless a doctor gives the okay. Avoid 'Agbo' or local liver cleansers."
            ),
            "high": "\n🚨 {subject} {verb_is} very ill with deep jaundice and confusion. Go to the hospital immediately for specialized liver care."
        }
    },


    "kidney stones": {
        "symptoms": [
            "strong pain for back and side",
            "pain dey enter front to private part",
            "piss dey sting",
            "blood for piss",
            "severe flank pain",
            "groin pain",
            "painful urination",
            "hematuria"
        ],
        "advice": {
            "low": (
                "\n{subject} should drink plenty of water to help flush the system."
                "\n\n🍎 **Diet Recommendation:** Drink plenty of water (at least 3 liters a day) and natural lemon water. The citrate in lemons can help prevent stone formation."
                "\n\n🛡️ **Precautions:** Stay active with light walking to help small stones pass. Monitor the color of {possessive} urine; it should be pale yellow, not dark."
                "\n\n🚫 **Avoid:** Avoid excessive salt (maggi/salt) and high-oxalate foods like spinach (Ugu) or peanuts if {subject} is prone to stones. Limit bottled sodas."
            ),
            "moderate": (
                "\nThis could be kidney stones. {subject} {verb_needs} an ultrasound scan and proper pain medication."
                "\n\n🥗 **Diet Recommendation:** Reduce intake of animal protein (heavy meat). Focus on plant-based proteins like beans in moderate amounts."
                "\n\n🛡️ **Precautions:** If {subject} manages to pass a stone in the urine, try to keep it in a clean container to show the doctor for analysis."
                "\n\n🚫 **Avoid:** Do not take high doses of Vitamin C supplements, as these can increase stone risk in some people. Avoid 'dry' periods without drinking water."
            ),
            "high": "\n🚨 {subject} {verb_is} in extreme pain, vomiting, or cannot pass urine at all. Go to the hospital immediately; the stone may be blocking the urinary tract."
        }
    },


    "anemia": {
        "symptoms": [
            "face white",
            "body weak all the time",
            "fast heart beat",
            "shortness of breath",
            "pale skin",
            "fatigue",
            "palpitations",
            "dizziness",
            "short breath"
        ],
        "advice": {
            "low": (
                "\n{subject} should eat more iron-rich foods like green vegetables and liver."
                "\n\n🍎 **Diet Recommendation:** Increase intake of 'blood-building' foods: fluted pumpkin leaves (Ugu), liver, kidneys, beans, and eggs."
                "\n\n🛡️ **Precautions:** Take Vitamin C (oranges/limes) alongside iron-rich meals to help the body absorb the iron better."
                "\n\n🚫 **Avoid:** Avoid drinking tea or coffee immediately after a meal, as the tannins can block iron absorption."
            ),
            "moderate": (
                "\n{possessive} blood level (PCV) might be low. {subject} {verb_needs} a blood test and iron supplements."
                "\n\n🥗 **Diet Recommendation:** Eat iron-fortified cereals and dark leafy greens. Consider 'Zobo' (Hibiscus tea) without too much sugar as a healthy blood-supporting drink."
                "\n\n🛡️ **Precautions:** Stand up slowly from a sitting or lying position to avoid dizziness (orthostatic hypotension)."
                "\n\n🚫 **Avoid:** Avoid heavy physical exertion until {possessive} blood levels improve. Do not rely solely on 'malt and milk'—while popular, it is not a substitute for medical iron supplements."
            ),
            "high": "\n🚨 {subject} {verb_is} severely anemic, fainting, or struggling to breathe while resting. {subject} may need an urgent blood transfusion. Go to the hospital now."
        }
    },

    "epilepsy/seizure": {
        "symptoms": [
            "body dey shake all over",
            "no fit talk or respond",
            "foam for mouth",
            "person fall down suddenly",
            "convulsion",
            "shaking of limbs",
            "unconsciousness",
            "loss of control",
            "frothing at mouth"
        ],
        "advice": {
            "low": (
                "\nEnsure {subject} gets enough sleep and avoids triggers. Monitor any small 'absent' moments."
                "\n\n🍎 **Diet Recommendation:** Maintain a steady, healthy diet to avoid low blood sugar, which can trigger seizures in some people."
                "\n\n🛡️ **Precautions:** Ensure the home environment is safe (no sharp corners). Make sure friends and family know how to perform seizure first aid (side-lying position)."
                "\n\n🚫 **Avoid:** Avoid flickering lights (strobe lights) and extreme sleep deprivation. Avoid high-stress environments where possible."
            ),
            "moderate": (
                "\n{subject} {verb_needs} to see a neurologist to start daily medication to prevent future fits."
                "\n\n🥗 **Diet Recommendation:** Some people benefit from a ketogenic diet (high fat, low carb), but this must only be done under strict medical supervision."
                "\n\n🛡️ **Precautions:** Carry a medical ID card or bracelet stating the condition. Never swim alone; always have someone nearby who knows about the condition."
                "\n\n🚫 **Avoid:** Do not stop taking seizure medication suddenly, even if {subject} hasn't had a fit in a long time. Avoid alcohol, as it interferes with medication."
            ),
            "high": "\n🚨 The seizure has lasted more than 5 minutes, or {subject} is having one fit after another without waking up. Seek emergency care immediately."
        }
    },

    "menstrual pain (dysmenorrhea)": {
        "symptoms": [
            "bad pain for lower belly during period",
            "back pain during period",
            "pain for waist",
            "period pain too much",
            "cramping",
            "pelvic pain",
            "lower back ache"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} mild menstrual pain. {subject} should use a hot water bottle on {possessive} belly and rest."
                "\n\n🍎 **Diet Recommendation:** Eat bananas and dark chocolate to help with cramps. Ginger or chamomile tea can also relax the uterine muscles."
                "\n\n🛡️ **Precautions:** Light exercise like walking or stretching can actually help reduce pain by increasing blood flow."
                "\n\n🚫 **Avoid:** Reduce salt and caffeine intake a few days before {possessive} period to reduce bloating and tension."
            ),
            "moderate": (
                "\nThis looks like dysmenorrhea. {subject} {verb_needs} pain medicine like paracetamol. If the pain no gree go, see a doctor."
                "\n\n🥗 **Diet Recommendation:** Take Magnesium and Vitamin B1 rich foods like beans, nuts, and whole grains to help reduce the severity of cramps."
                "\n\n🛡️ **Precautions:** Track the cycle to see if the pain is getting worse over time. Heavy pain can sometimes indicate underlying issues like fibroids."
                "\n\n🚫 **Avoid:** Avoid very cold water or 'iced' drinks if {subject} finds they make the cramping feel more intense."
            ),
            "high": "\n⚠️ {subject} {verb_is} in severe, debilitating pain. If {subject} {verb_is} fainting, vomiting from pain, or soaking through pads in less than an hour, please go to the hospital immediately."
        }
    },

    "preeclampsia": {
        "symptoms": [
            "BP high during pregnancy",
            "headache no gree go",
            "eye no clear",
            "face and leg dey swell fast",
            "hypertension in pregnancy",
            "severe headache",
            "blurred vision",
            "sudden swelling (edema)"
        ],
        "advice": {
            "low": (
                "\n⚠️ Note, {subject} must monitor {possessive} blood pressure daily. Any increase is a danger sign."
                "\n\n🍎 **Diet Recommendation:** Eat a balanced diet rich in calcium (yogurt, sardines, green vegetables). Drink enough water."
                "\n\n🛡️ **Precautions:** Rest on {possessive} left side to improve blood flow to the baby and kidneys. Attend every antenatal appointment without fail."
                "\n\n🚫 **Avoid:** Avoid adding extra salt to food. Do not use 'herbal' pregnancy cleansers, as these can dangerously spike blood pressure."
            ),
            "moderate": (
                "\n{subject} {verb_has} signs of preeclampsia. {subject} {verb_needs} to see {possessive} doctor today for a check-up."
                "\n\n🥗 **Diet Recommendation:** Focus on high-protein foods and fiber. Avoid processed snacks and canned foods that are high in sodium."
                "\n\n🛡️ **Precautions:** Watch for 'warning signs' like seeing spots/stars or pain just below the ribs. Reduce physical activity and stress immediately."
                "\n\n🚫 **Avoid:** Avoid high-stress situations and long hours of standing. Do not skip blood pressure medication if it has been prescribed by a doctor."
            ),
            "high": "\n🚨 {subject} {verb_is} showing critical signs like blurred vision, severe headache, or swelling of the face. Rush to the hospital now; this is a life-threatening emergency for both {subject} and the baby."
        }
    },

    "fracture/broken bone": {
        "symptoms": [
            "strong pain for hand or leg when e break",
            "e no fit move the place",
            "place dey look crooked",
            "swelling and bruising",
            "severe pain",
            "inability to use limb",
            "deformity"
        ],
        "advice": {
            "low": (
                "\nThis could be a minor bone injury. {subject} should rest and keep the limb still."
                "\n\n🍎 **Diet Recommendation:** Eat foods high in Calcium and Vitamin D (milk, eggs, fish) to support bone repair."
                "\n\n🛡️ **Precautions:** Use the 'RICE' method: Rest, Ice (if available), Compression (wrap lightly), and Elevation (keep the limb raised)."
                "\n\n🚫 **Avoid:** Do not try to 'massage' or 'pull' the limb back into place yourself, as this can cause more damage."
            ),
            "moderate": (
                "\nThis could be a fractured or broken bone. {subject} should use a splint to keep the place straight and go to the hospital."
                "\n\n🥗 **Diet Recommendation:** Focus on protein-rich meals to help build the 'scaffold' the bone needs to heal. Avoid alcohol, which slows down bone healing."
                "\n\n🛡️ **Precautions:** If the skin is broken near the fracture, cover it with a clean cloth to prevent infection. Keep the limb immobilized (not moving) until seen by a doctor."
                "\n\n🚫 **Avoid:** Avoid going to 'traditional bone setters' first; get an X-ray at the hospital to ensure the bone is aligned correctly so it doesn't heal crooked."
            ),
            "high": "\n🚨 This is a severe fracture (bone may be sticking out). Do not move {subject}. Stop any heavy bleeding with a clean cloth and call for emergency transport immediately."
        }
    },

    "severe allergic reaction (anaphylaxis)": {
        "symptoms": [
            "skin dey scratch and swell fast",
            "throat tight, no fit breathe",
            "tongue swell",
            "fast heart beat",
            "difficulty breathing",
            "hives",
            "swelling of lips/tongue/throat",
            "wheezing",
            "collapse"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} an allergy. {subject} should take an antihistamine and stay away from the cause."
                "\n\n🍎 **Diet Recommendation:** Stick to simple, non-processed foods while the reaction subsides. Drink water to stay hydrated."
                "\n\n🛡️ **Precautions:** Identify exactly what caused the reaction (food, insect bite, soap) and write it down. Keep an antihistamine (like Loratadine) in {possessive} first aid kit."
                "\n\n🚫 **Avoid:** Avoid the trigger immediately. If it was food, do not eat any more of it, even a small bite."
            ),
            "moderate": (
                "\n{possessive} allergy is worsening. {subject} {verb_needs} medical attention before the throat closes."
                "\n\n🥗 **Diet Recommendation:** Avoid spicy or acidic foods that might further irritate the mouth or throat during a reaction."
                "\n\n🛡️ **Precautions:** Sit upright to make breathing easier. If the reaction is to a sting, remove the stinger carefully without squeezing it."
                "\n\n🚫 **Avoid:** Do not wait to see if it 'goes away' if {subject} feels itchy in the throat or starts wheezing. Do not panic, as fast heart rates can make the reaction feel worse."
            ),
            "high": "\n🚨 {subject} {verb_is} in anaphylactic shock and cannot breathe well. This is life-threatening. Rush to the hospital for an adrenaline injection immediately!"
        }
    },

    "schistosomiasis (bilharzia)": {
        "symptoms": [
            "blood for piss",
            "stomach dey pain (lower part)",
            "blood for stool",
            "frequent urination",
            "hematuria",
            "abdominal pain",
            "bloody stool",
            "bladder pain"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} signs of Bilharzia. {subject} should avoid swimming in fresh water (rivers/ponds) and get a urine test."
                "\n\n🍎 **Diet Recommendation:** Eat a balanced diet to support the immune system. Stay hydrated to help with urinary discomfort."
                "\n\n🛡️ **Precautions:** Only use water that has been boiled or filtered for bathing and washing. Warn others in the community about the water source."
                "\n\n🚫 **Avoid:** Do not enter fresh water rivers, ponds, or lakes where snails might live. Avoid walking barefoot in damp soil near these water bodies."
            ),
            "moderate": (
                "\nThis could be Schistosomiasis. {subject} {verb_needs} proper deworming medicine (Praziquantel) from a health center."
                "\n\n🥗 **Diet Recommendation:** Eat iron-rich foods (liver, Ugu) as the infection can cause blood loss and anemia over time."
                "\n\n🛡️ **Precautions:** Complete the full course of medication. Ensure {subject} has a follow-up test in a few weeks to make sure the parasites are gone."
                "\n\n🚫 **Avoid:** Avoid self-treating with regular dewormers bought on the street; they often do not kill Schistosoma parasites. You need the specific hospital-grade dose."
            ),
            "high": "\n🚨 Heavy blood in the urine or stool indicates a severe infection or damage to the bladder/intestines. {subject} must visit the hospital for a full check-up and organ scan."
        }
    },

    "gallstones": {
        "symptoms": [
            "sharp pain for upper right belly",
            "pain after chop oily food",
            "yellow eye",
            "vomiting",
            "severe right upper quadrant pain",
            "pain radiating to back/shoulder",
            "jaundice"
        ],
        "advice": {
            "low": (
                "\n{subject} may have gallstones. {subject} should avoid oily and fatty foods to prevent the pain from returning."
                "\n\n🍎 **Diet Recommendation:** Switch to a low-fat diet. Eat more fiber (whole grains, vegetables). Drink plenty of water."
                "\n\n🛡️ **Precautions:** Maintain a healthy weight, but avoid 'crash dieting' or losing weight too fast, as this can actually cause more gallstones."
                "\n\n🚫 **Avoid:** Avoid fried foods (fried meat, puff-puff, akara), butter, and heavy cream. Limit red meat."
            ),
            "moderate": (
                "\nThis looks like a gallbladder issue. {subject} {verb_needs} an abdominal scan at a clinic to confirm if stones are present."
                "\n\n🥗 **Diet Recommendation:** Focus on lean proteins like skinless chicken or fish. Small, frequent meals are better than one large, heavy meal."
                "\n\n🛡️ **Precautions:** If {subject} feels a 'gallbladder attack' coming on (pain after eating), try to stay calm and sit upright. Keep a record of which foods trigger the pain."
                "\n\n🚫 **Avoid:** Avoid highly processed snacks and 'junk' food. Do not ignore the pain even if it goes away, as the stones are still there."
            ),
            "high": "\n🚨 If {subject} {verb_is} in intense pain, vomiting, or {possessive} eyes are yellow (jaundice), the stone might be blocking a bile duct. Go to the hospital now; this can lead to a serious infection."
        }
    },

    "dengue fever": {
        "symptoms": [
            "very bad headache",
            "pain behind eye",
            "joint and muscle pain too much",
            "rash",
            "high fever",
            "severe bone pain (breakbone fever)",
            "retro-orbital pain",
            "skin rash"
        ],
        "advice": {
            "low": (
                "\n{subject} likely {verb_has} Dengue. {subject} should rest and take only Paracetamol. Do NOT take Ibuprofen or Aspirin."
                "\n\n🍎 **Diet Recommendation:** Drink plenty of fluids—water, fruit juices, and coconut water—to prevent dehydration from the high fever."
                "\n\n🛡️ **Precautions:** Use mosquito nets and repellent to prevent further bites and avoid spreading the virus to others in the house."
                "\n\n🚫 **Avoid:** **STRICTLY AVOID** Ibuprofen (Aboliki, Advil), Aspirin, or Diclofenac. These can increase the risk of internal bleeding with Dengue."
            ),
            "moderate": (
                "\nThis looks like Dengue fever. {subject} {verb_needs} to stay very hydrated and be monitored by a health worker for any bleeding signs."
                "\n\n🥗 **Diet Recommendation:** Eat light, vitamin-rich foods like fruit purees or vegetable soups to keep {possessive} strength up."
                "\n\n🛡️ **Precautions:** Monitor the skin for tiny red purple spots (petechiae). Watch for any bleeding from the gums when brushing teeth."
                "\n\n🚫 **Avoid:** Avoid heavy manual labor. Do not go to a chemist to get 'injections' unless they know it's Dengue, as some injections can cause bleeding at the site."
            ),
            "high": "\n🚨 If {subject} starts bleeding from the gums or nose, has severe stomach pain, or is vomiting blood, it is Dengue Hemorrhagic Fever. Seek emergency care immediately."
        }
    },

    "stroke": {
        "symptoms": [
            "one side of face drop",
            "one hand or leg no fit move",
            "trouble talk",
            "speech difficulty"
            "confusion (confuse)"
            "sudden onset of weakness",
            "facial drooping",
            "slurred speech",
            "loss of balance",
            "stroke",
            "severe headache"
        ],
        "advice": {
            "low": (
                "\n⚠️ Even if symptoms are low, {subject} {verb_needs} a brain scan immediately to prevent a full stroke."
                "\n\n🍎 **Diet Recommendation:** Follow a heart-healthy diet. Eat plenty of fiber from vegetables and oats. Oily fish like Mackerel (Titus) can help with blood flow."
                "\n\n🛡️ **Precautions:** Monitor blood pressure daily. If {subject} feels any sudden numbness or 'pins and needles' on one side, seek help instantly."
                "\n\n🚫 **Avoid:** Avoid high-salt foods and cubes with high MSG, which can spike blood pressure. Stop smoking and avoid secondhand smoke entirely."
            ),
            "moderate": (
                "\n{subject} {verb_is} showing signs of a stroke. Do not wait. Take {subject} to the hospital right now."
                "\n\n🥗 **Diet Recommendation:** Focus on 'Low-Sodium' meals. Use natural spices like ginger, garlic, and onions instead of salt to flavor food."
                "\n\n🛡️ **Precautions:** Keep the patient calm and sitting or lying down while waiting for transport. Note the exact time the symptoms started; this is vital for doctors."
                "\n\n🚫 **Avoid:** Do not give {subject} any food, water, or aspirin until a doctor confirms it is safe, as swallowing may be difficult and can cause choking."
            ),
            "high": "\n🚨 {subject} {verb_is} having a major stroke. Every minute counts to save {possessive} brain. Rush to the nearest Emergency Center immediately!"
        }
    },

    "appendicitis": {
        "symptoms": [
            "pain start for navel and move to lower right belly",
            "vomiting",
            "fever",
            "stomach pain for right corner",
            "loss of appetite",
            "periumbilical pain",
            "right lower quadrant pain",
            "rebound tenderness"
        ],
        "advice": {
            "low": (
                "\n{subject} should monitor the pain. If it moves to the lower right side, it could be the appendix."
                "\n\n🍎 **Diet Recommendation:** Stick to very light, clear liquids like water or light broth if {subject} feels nauseous."
                "\n\n🛡️ **Precautions:** Rest in a comfortable position. Avoid any heavy lifting or straining the stomach muscles."
                "\n\n🚫 **Avoid:** Avoid eating heavy or solid meals until the cause of the pain is known. Do not take laxatives or 'purging' medicines, as these can cause the appendix to burst."
            ),
            "moderate": (
                "\nThis may be appendicitis. {subject} should go to the hospital for a scan."
                "\n\n🥗 **Diet Recommendation:** Stop eating and drinking entirely (NPO) once the pain becomes localized to the right side, in case surgery is needed."
                "\n\n🛡️ **Precautions:** If the pain suddenly disappears, do not assume it is cured; this can actually mean the appendix has ruptured. See a doctor immediately."
                "\n\n🚫 **Avoid:** Avoid applying heat (like a hot water bottle) to the stomach, as heat can increase the risk of inflammation and rupture."
            ),
            "high": "\n🚨 The appendix may burst. {subject} {verb_needs} surgery immediately. Go to the hospital emergency room now."
        }
    },

    "lassa fever": {
        "symptoms": [
            "high fever wey no gree go",
            "body pain",
            "throat pain",
            "face swell",
            "bleeding from gums/nose",
            "severe fever",
            "facial swelling",
            "sore throat",
            "hemorrhage"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} a persistent fever. Monitor {possessive} symptoms very closely."
                "\n\n🍎 **Diet Recommendation:** Stay hydrated with clean, bottled water and fruit juices (orange or pineapple) to support the immune system."
                "\n\n🛡️ **Precautions:** Practice strict hand hygiene. Cover all food storage containers to prevent contact with rats (mastomys) or their droppings."
                "\n\n🚫 **Avoid:** Avoid eating any food that may have been in contact with rodents. Do not dry food (like garri or yam) in the open where rats can reach it."
            ),
            "moderate": (
                "\nThis could be Lassa Fever. {subject} {verb_needs} to go to the nearest health center immediately for isolation and treatment."
                "\n\n🥗 **Diet Recommendation:** Eat soft, high-energy foods like pap or mashed yams. Keep fluids high to prevent dehydration from the fever."
                "\n\n🛡️ **Precautions:** Family members should avoid contact with {subject}'s body fluids (blood, urine, saliva). Use gloves if care is necessary."
                "\n\n🚫 **Avoid:** Do not treat this as 'normal malaria'. Avoid going to crowded places to prevent spreading the virus to others."
            ),
            "high": "\n🚨 This is a life-threatening infection. {subject} {verb_is} showing severe Lassa Fever signs. Seek specialized medical help immediately."
        }
    },
    "hemorrhoids (piles)": {
        "symptoms": [
            "pain and itching for anus (piles)",
            "blood when you stool (small blood)",
            "lump for outside anus",
            "painful bowel movement",
            "rectal bleeding",
            "anal itching",
            "prolapse"
        ],
        "advice": {
            "low": (
                "\n{subject} may have hemorrhoids or piles. {subject} should eat high-fiber food and drink lots of water."
                "\n\n🍎 **Diet Recommendation:** Increase fiber intake significantly. Eat plenty of beans, brown rice, 'Ofada' rice, and fruits like pawpaw and bananas."
                "\n\n🛡️ **Precautions:** Try a 'Sit bath'—sit in warm water for 10–15 minutes several times a day to reduce swelling and pain."
                "\n\n🚫 **Avoid:** Avoid sitting for long periods on the toilet. Do not use dry or rough toilet paper; use moist wipes or water instead."
            ),
            "moderate": (
                "\nFor these piles, {subject} should avoid straining when stooling. See a doctor for ointment."
                "\n\n🥗 **Diet Recommendation:** Drink at least 8–10 glasses of water daily. Prune juice or warm water with lemon in the morning can help soften stool."
                "\n\n🛡️ **Precautions:** Use over-the-counter stool softeners if recommended by a pharmacist. Avoid lifting very heavy objects which increases pressure on the area."
                "\n\n🚫 **Avoid:** Avoid spicy foods (pepper) and alcohol, which can irritate the digestive tract and make bowel movements more painful."
            ),
            "high": "\n⚠️ If the pain is severe or the bleeding is heavy, {subject} {verb_needs} to see a doctor for possible surgery or advanced care."
        }
    },

    "gonorrhea (std)": {
        "symptoms": [
            "yellow or white discharge from private part",
            "pain when you piss",
            "testicles dey swell for men",
            "vaginal/penile discharge",
            "dysuria",
            "pelvic inflammatory disease (PID)",
            "epididymitis"
        ],
        "advice": {
            "low": (
                "\n{subject} may have an infection. {subject} should get a test at a clinic soon."
                "\n\n🍎 **Diet Recommendation:** Eat probiotic-rich foods like plain unsweetened yogurt to help balance the body's natural bacteria."
                "\n\n🛡️ **Precautions:** Drink plenty of water to help flush the urinary tract. Inform any recent sexual partners so they can also get tested."
                "\n\n🚫 **Avoid:** Avoid all sexual activity until you have been cleared by a doctor. Do not try to 'wash' the discharge away with harsh chemicals or soaps inside the private area."
            ),
            "moderate": (
                "\nThis looks like Gonorrhea (STD). {subject} {verb_needs} a doctor for a test and antibiotics."
                "\n\n🥗 **Diet Recommendation:** Eat foods high in Vitamin C (citrus fruits, bell peppers) to boost the immune system's ability to fight the infection."
                "\n\n🛡️ **Precautions:** Complete the entire course of antibiotics prescribed by the doctor, even if symptoms disappear after one or two days."
                "\n\n🚫 **Avoid:** Avoid self-medicating with random 'G-wash' or herbal drinks, as untreated gonorrhea can lead to infertility."
            ),
            "high": "\n⚠️ The infection may be spreading. {subject} must see a doctor immediately to prevent long-term damage like PID or chronic pelvic pain."
        }
    },

    "syphilis (std)": {
        "symptoms": [
            "painless sore for private part",
            "rash for hand and leg bottom",
            "sore for mouth",
            "painless chancre",
            "non-itchy rash",
            "fever",
            "lymphadenopathy"
        ],
        "advice": {
            "low": (
                "\n{subject} {verb_has} symptoms that could be an early STD like Syphilis. {subject} should get a blood test soon."
                "\n\n🍎 **Diet Recommendation:** Focus on a balanced diet rich in leafy greens and lean proteins (fish/beans) to maintain strength during treatment."
                "\n\n🛡️ **Precautions:** Be aware that the painless sore (chancre) will heal on its own, but the infection is still in the body. Testing is the only way to be sure."
                "\n\n🚫 **Avoid:** Avoid sexual contact until a doctor confirms the infection is treated. Do not apply creams or ointments to the sore without medical advice."
            ),
            "moderate": (
                "\nThis looks like Syphilis. {subject} {verb_needs} a penicillin injection from a clinic."
                "\n\n🥗 **Diet Recommendation:** Stay hydrated. Garlic and ginger can be added to meals for their general anti-inflammatory and immune-boosting properties."
                "\n\n🛡️ **Precautions:** Ensure your partner is treated at the same time, or you will simply catch it again after your treatment."
                "\n\n🚫 **Avoid:** Avoid alcohol during the treatment phase, as your body needs to focus on clearing the infection and responding to the medication."
            ),
            "high": "\n🚨 Syphilis can affect the brain and heart if left too long. Since {subject} {verb_is} showing advanced signs (like rashes or neurological changes), see a specialist today."
        }
    },

    "hypoglycemia (low blood sugar)": {
        "symptoms": [
            "body dey shake",
            "sweating",
            "dizzy",
            "confusion",
            "feel like fainting",
            "tremors",
            "palpitations",
            "anxiety",
            "fainting"
        ],
        "advice": {
            "low": (
                "\nIt could be that {possessive} sugar level is low. {subject} should quickly eat something sweet."
                "\n\n🍎 **Diet Recommendation:** Follow the '15-15 Rule': Eat 15 grams of fast-acting sugar (like 3 sugar cubes, half a cup of juice, or a tablespoon of honey) and wait 15 minutes."
                "\n\n🛡️ **Precautions:** Always carry a small piece of candy or a sugar packet if {subject} is prone to low sugar. Check blood sugar immediately if a meter is available."
                "\n\n🚫 **Avoid:** Avoid heavy exercise when sugar is low. Do not skip meals, especially if taking diabetes medication."
            ),
            "moderate": (
                "\nThis is hypoglycemia. {subject} {verb_needs} to eat a proper meal now and check {possessive} sugar level."
                "\n\n🥗 **Diet Recommendation:** After the initial sugar boost, eat a snack with complex carbohydrates and protein (like crackers with peanut butter or a small bowl of beans) to keep sugar steady."
                "\n\n🛡️ **Precautions:** If {subject} {verb_is} diabetic, discuss these episodes with a doctor; the medication dosage may need to be adjusted."
                "\n\n🚫 **Avoid:** Avoid eating 'complex' foods like chocolate or cake to treat an initial low, as the fat in them slows down the sugar absorption when you need it fast."
            ),
            "high": "\n🚨 {subject} {verb_is} at risk of a coma. Give {subject} sugar immediately. If {subject} is unconscious, do not put food in the mouth; rush to the hospital for a glucose drip."
        }
    },

    "glaucoma (eye pressure)": {
        "symptoms": [
            "eye pain strong",
            "headache bad",
            "eye no see clear fast (can't see well, patchy blind)",
            "eye red",
            "severe eye pain",
            "sudden blurred vision",
            "seeing halos ( seeing circle of light)",
            "redness of eye"
        ],
        "advice": {
            "low": (
                "\n{subject} should have {possessive} eye pressure checked by an eye doctor soon."
                "\n\n🍎 **Diet Recommendation:** Eat foods high in antioxidants (blueberries, leafy greens) and Vitamin C. Some studies suggest hot tea may lower the risk of glaucoma."
                "\n\n🛡️ **Precautions:** Avoid activities that put pressure on the eyes, such as hanging the head down for long periods or wearing very tight neckties."
                "\n\n🚫 **Avoid:** Avoid smoking, as it increases eye pressure. Do not use 'over-the-counter' steroid eye drops without a prescription, as these can cause glaucoma."
            ),
            "moderate": (
                "\nThis may be Glaucoma. High pressure can blind {subject} fast. {subject} {verb_needs} to see an ophthalmologist today."
                "\n\n🥗 **Diet Recommendation:** Maintain a healthy weight and lower insulin levels by reducing sugary foods and refined flour (white bread/white rice)."
                "\n\n🛡️ **Precautions:** If prescribed eye drops, {subject} must use them at the exact same time every day without fail. They are 'life-savers' for vision."
                "\n\n🚫 **Avoid:** Avoid drinking large amounts of water very quickly (more than a liter in minutes), as this can temporarily raise eye pressure."
            ),
            "high": "\n🚨 Sudden vision loss, halos around lights, or severe eye pain with vomiting is an emergency. Go to an eye clinic right now."
        }
    }
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
    "BP don high", "BP high", "BP high during pregnancy", "I can't pee", "I cough blood",
    "I dey feel very weak", "I dey hawk phlegm", "I dey perspire for night", "I dey run to toilet always",
    "I dey scratch every time", "I dey see black spot", "I dey see double", "I dey slim down fast",
    "I dey struggle to breathe", "I dey throw up in the morning", "I dey tire quick",
    "I feel like say I go fall down", "I feel sand sand inside my eye", "I feel tired quickly",
    "abdominal cramps", "abdominal discomfort", "abdominal mass", "abdominal pain", "ache all over",
    "aching legs", "aching limbs", "acid reflux", "acute joint pain (toe)", "always hungry (too much hunger)",
    "always thirsty", "always want to eat", "amenorrhea", "anal itching", "anhedonia", "ankle swelling",
    "anxiety", "arrhythmia", "athlete's foot", "back pain", "back pain during period", "back stiffness",
    "bad itching for night time", "bad pain for lower belly during period", "belly big", "belly dey grumble",
    "belly full of wind", "big fever", "bleeding from gums/nose", "blisters",
    "blisters (skin get water blister)", "bloating", "blood for piss", "blood for stool", "blood in urine",
    "blood when you stool (small blood)", "bloody stool", "blurred vision", "blurred vision (eye no see clear)",
    "blurred vision (no see clear)", "blurry", "body aches (body pain)", "body dey burn", "body dey shake all over",
    "body dey dry up", "body dey pain everywhere", "body dey pain me well well", "body dey waste", "body thin",
    "body weak", "body weak all the time", "body weak die", "body weak always", "bone pain", "breast dey heavy",
    "breast tenderness (breast dey pain)", "breath no dey enter well", "breathing dey sound",
    "burning during urination", "burning eyes", "burning pain for stomach (pain for stomach like fire)",
    "burning sensation", "cachexia", "can't see far", "catarrh", "chest dey hold me", "chest dey make noise",
    "chest pain (chest dey pain or tight)", "chest pain (chest dey pain)", "chest pain when I cough",
    "chest tightness", "child dey small", "child dey thin", "child no grow well", "chills",
    "chills (body dey shake)", "chronic cough", "chronic diarrhea", "circular rash", "cluster headache",
    "cold dey catch me", "collapse", "confusion", "conjunctivitis", "constant sadness", "constant sneezing",
    "constipation", "constipation (no fit stool)", "convulsion", "coryza (runny nose)", "cough",
    "cough (dey cough mostly for night)", "cough (dey cough)", "cough for long time", "cough for night",
    "cough wey last more than 2 weeks", "cough wey no end", "cough with mucus (cough dey bring phlegm)",
    "cough, runny nose, red eyes", "coughing blood (cough dey bring blood)", "coughing out yellow tin",
    "cramping", "cravings", "dark piss", "dark tarry stool", "debilitating weakness", "deformity",
    "dehydration (mouth dry)", "dermatitis", "dey beat my head", "dey chop ice/dirt (pica)", "dey gasp for air",
    "dey piss small small", "dey pour water", "dey purge", "dey purge steady", "dey vomit food", "diarrhea",
    "diarrhea (stool dey run)", "diarrhea (stool dey rush)", "difficulty breathing",
    "difficulty breathing (no fit breathe well)", "difficulty standing straight", "discharge", "dizziness",
    "dizziness (head dey turn)", "double vision", "drenching night sweats", "drenching sweat",
    "dry mouth (tongue dry)", "dry throat always", "dry tongue", "dyspareunia", "dyspnea on exertion",
    "dysuria", "e don pass my time", "earache", "edema", "e no fit move the place", "enlarged, twisted veins",
    "epigastric burning pain", "epistaxis", "everywhere dey pain", "excess body fat",
    "excessive thirst (always dey thirsty)", "excessive thirst (mouth dey dry)", "exhaustion",
    "extreme fatigue", "extreme thirst", "eye dey burn", "eye dey confuse me", "eye dey sticky",
    "eye dey tear up", "eye dey tear with light", "eye full of pus", "eye mattering", "eye no clear",
    "eye no see clear fast", "eye pain (eye dey pain)", "eye pain strong", "eye red", "eye red like fire",
    "face and leg dey swell fast", "face swell", "face swell for side (like big cheek)", "face white",
    "facial drooping", "facial swelling", "fainting", "fast breathing (dey breathe fast)", "fast heart beat",
    "fast or irregular heartbeat (heart dey beat fast)", "fatigue", "fatigue (body weak)",
    "fatigue (no strength)", "feel faint", "feel full quick (belly full fast)", "feeling cold and hot",
    "fever", "fever (body hot / body dey hot/high temperature)", "fever (body hot)", "fever and cold",
    "fever high", "fever too high o", "fever wey last many days(fever)", "food aversions",
    "food no dey sweet me", "foam for mouth", "frequent urge to urinate (dey always wan piss)",
    "frequent urination", "frequent urination (dey piss too much)", "frequent watery stools",
    "frothing at mouth", "gallstones", "gasping for air", "general malaise", "general weakness",
    "gestation", "glaucoma", "groin pain", "groin swelling", "gritty feeling", "hard to breathe",
    "head dey big for small body", "head dey burst", "head heavy", "head hurts", "headache",
    "headache (head dey pain often)",
    "headache (head dey pain)", "headache bad", "heart dey beat fast", "heart dey beat quick", "heart dey jump",
    "heartburn", "hematuria", "hemoptysis", "hemorrhage", "heavy head", "high fever",
    "high fever (body dey hot well well)", "high fever (body hot well well)", "high fever wey no gree go",
    "high temperature", "hives", "hot body for weeks", "hot piss", "hotness for body", "hyperglycemia",
    "hypertension", "hypertension in pregnancy", "hyperuricemia (high acid)", "hypersomnia",
    "inability to use limb", "indigestion", "inguinal lump", "insomnia", "intense headache",
    "intense nocturnal itching", "itchy eye (eye dey scratch)", "itchy ring-shaped rash",
    "itchy skin (skin dey scratch)", "itchy rash (small water blister)", "jaundice", "jock itch",
    "joint and muscle pain too much", "joint dey swell up", "joint pain", "joint red and hot",
    "joint stiffness", "koplik spots (white spots in mouth)", "kwashiorkor", "laboured breathing",
    "leg dey heavy and pain after standing", "leg dey swell for evening", "leg dey swell up", "leg heaviness",
    "light sensitivity (eye no like light)", "light-headedness", "loss of appetite",
    "loss of appetite (no wan chop)", "loss of balance", "loss of control",
    "loss of interest in wetin you like", "loss of sight", "loss of taste", "loose motion",
    "long lasting fever (body hot for weeks)", "long term weakness", "lower back ache",
    "lower stomach pain (lower belly dey pain)", "lumbago", "lump dey disappear when you lie down",
    "lump dey show for groin or belly", "lump for outside anus", "lymph nodes swelling", "lymphadenopathy",
    "maculopapular rash", "malaise", "marasmus", "melena", "mild fever (small body hot)",
    "missed period (menstruation no come)", "morning sickness", "mouth dry finish", "mucus discharge",
    "muscle aches (body pain)", "muscle pain", "muscle stiff for back", "musculoskeletal pain",
    "my body is vibrating", "my bone dey pain", "my brain dey knock", "my eye dey bring sand",
    "my hand dey weak", "my head dey heavy with catarrh", "my head dey turn", "my head dey turn me",
    "my sight don weak", "my skin dey red", "my throat dey wound me", "nasal congestion (nose block)",
    "nausea", "nausea (body dey turn)", "neck dey swell", "neck no gree bend", "need to piss always",
    "nervousness", "night sweats (body dey sweat for night)", "no fit chop anything",
    "no fit talk or respond", "no solid stool", "no strength for child", "no urine (no dey piss)",
    "no wan talk to anybody", "non-itchy rash", "nose dey leak water", "nuchal rigidity", "obesity",
    "occipital headache", "oliguria", "one hand or leg no fit move", "one side of face drop",
    "pain after chop oily food", "pain after eating (pain after you chop)", "pain and itching for anus (piles)",
    "pain better after eating", "pain behind eye", "pain dey enter front to private part",
    "pain dey shoot to leg", "pain during sex", "pain for back", "pain for chest", "pain for head no gree stop",
    "pain for my chest like fire", "pain for my waist", "pain for upper belly like fire",
    "pain for the lump when you cough", "pain for waist wey no gree go",
    "pain for private part and between fingers", "pain inside my private area",
    "pain radiating to back/shoulder", "pain start for navel and move to lower right belly",
    "pain when you piss", "pain when you swallow", "pain when urinating (when you piss e dey pain)",
    "pain worse when hungry", "painful bowel movement", "painful swallowing", "painless chancre",
    "painless sore for private part", "pale skin", "palpitations", "papules and burrows",
    "parotid gland swelling", "pelvic inflammatory disease (PID)", "pelvic pain", "peptic ulcer disease",
    "period", "period no come this month", "period pain too much", "periumbilical pain",
    "persistent diarrhea (stool no dey stop)", "person dey fall down suddenly", "person dey slim down",
    "perspire for night", "phlegm", "photophobia", "pica (ice/dirt craving)", "piles", "piss dey sting",
    "piss too much", "pissing dey burn me", "place dey look crooked", "pleuritic chest pain", "pneumonia",
    "polyphagia", "polydipsia", "polyuria", "poor appetite", "pressure in head", "productive cough",
    "profuse night sweats", "prolapse", "prolonged fever", "pruritic vesicular rash", "pruritus",
    "purulent discharge", "rales", "rash", "rash (skin get small small spots)",
    "rash for face and spread down", "rash for hand and leg bottom", "rash for private part",
    "rash spread all over body", "rash in skin folds", "rebound tenderness", "rectal bleeding",
    "red eye (eye red)", "red for throat", "red spots (red mark for body)", "red spots on skin",
    "red, scaly skin patch", "redness of eye", "restlessness", "retro-orbital pain",
    "right lower quadrant pain", "running stomach", "runny nose (catarrh dey come out)",
    "sad all the time (no joy)", "sciatica", "severe allergic reaction (anaphylaxis)",
    "severe back and side pain", "severe body aches", "severe bone pain (breakbone fever)",
    "severe chills", "severe eye pain", "severe fever", "severe flank pain", "severe headache",
    "severe itching", "severe pain", "severe right upper quadrant pain", "shaking fever",
    "shaking of limbs", "sharp pain for upper right belly", "shivering", "shooting pain to leg",
    "short breath", "shortness of breath", "shortness of breath (no fit breathe well)",
    "shortness of breath when climbing stair", "skin burning", "skin dey vex me", "skin eruption",
    "skin rash", "sleep too much or too little", "slurred speech", "small fever", "small piss dey come",
    "small small bumps", "small small bumps or line on skin", "smallpox dey my body",
    "social withdrawal", "sore for mouth", "sore throat", "sore throat (throat dey pain)", "sore tongue",
    "soreness", "sometimes no symptoms at all", "spit blood", "speech Difficulty" "sticky discharge (eye gum when you wake)",
    "stiff neck", "stomachache", "stomach hurts", "stomach cramps (belly twist)", "stomach dey bleed",
    "stomach dey pain me bad",
    "stomach dey pain (lower part)", "stomach dey push me", "stomach dey run non-stop",
    "stomach dey worry me", "stomach pain", "stomach pain (belly dey pain)",
    "stomach pain for right corner", "stomach pain for right side", "stomach pain bad",
    "stomach rumbling", "stomach upset", "stooling too much", "strong body heat", "strong headache",
    "strong pain for back and side", "strong pain for hand or leg when e break",
    "strong pressure for inside head", "stunted growth", "stuffy nose",
    "sudden bad pain for toe (big toe)", "sudden blurred vision", "sudden onset of weakness",
    "sudden onset vomiting", "sudden swelling (edema)", "sugar dey high", "sugar level high",
    "sweat for night", "sweating", "sweating (excessive)", "sweating (too much sweat)",
    "swelling and bruising", "swelling in legs (leg dey swell)", "swelling of lips/tongue/throat",
    "swelling and redness", "swollen feet (leg dey swell from kwashiorkor)", "swollen tonsils",
    "syncope", "tender breasts", "tension headache", "testicles dey swell for men",
    "thin arms (hand dey thin)", "thick white discharge", "throat dey pain when I swallow",
    "throat dey scratch", "throat pain", "throat tight, no fit breathe", "throat irritation",
    "throbbing pain", "tired easily (body weak quick quick)", "tongue smooth and sore", "tongue swell",
    "too much piss", "too much weight gain", "tremors", "trouble sleeping", "trouble talk",
    "tummy dey burn", "tummy dey pain", "unconsciousness", "underweight", "unexplained weight loss",
    "unrelenting pain", "vaginal itching", "vaginal/penile discharge", "varicose veins",
    "veins for leg dey look big and twisted (like rope)", "vertigo", "very bad headache",
    "very thirsty", "vesicles", "visual impairment", "vision loss", "vision loss (eye dey blind small small)",
    "vomiting", "vomiting (dey vomit for morning)", "vomiting (dey vomit)", "vomiting blood (dey vomit blood)",
    "vomiting and diarrhea start quick after chop", "wasting", "wasting syndrome", "watery diarrhea",
    "watery eye (eye dey bring water)", "weakness", "weakness (body weak well well)", "weakness (body weak)",
    "weakness (no strength)", "weight loss (body dey reduce)", "weight loss (body dey slim down)",
    "weight loss (body too slim)", "wetin I dey swallow dey pain", "wheezing",
    "white thick discharge for private part (like cheese)", "widespread pain", "wound no dey close",
    "wounds no dey heal quick", "yellow eye", "yellow eye and skin", "yellow urine",
    "yellow or white discharge from private part"
]


def has_medical_relevance(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in medical_keywords)


train_ml_model()


def find_nearby_hospitals(user_lat, user_lon, radius_km=100):
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

    AVG_SPEED_KMH = 40
    # Retrieve all hospital records from the database.
    for hospital in Hospital.query.all():
        # Ensure the hospital has valid latitude and longitude coordinates.
        if hospital.lat is not None and hospital.lon is not None:
            # Calculate the Euclidean distance between the user and the hospital's coordinates.
            # Approximation: 1 degree latitude ~ 111 km. Rough for longitude, especially away from equator.
            distance_in_degrees = ((user_lat - hospital.lat) ** 2 + (user_lon - hospital.lon) ** 2) ** 0.5
            distance_in_km = round(distance_in_degrees * 111, 1)
            # If the hospital is within the specified radius, add it to the nearby list.
            if distance_in_km <= radius_km:
                # Calculate estimated travel time in minutes
                travel_time_min = round((distance_in_km / AVG_SPEED_KMH) * 60)

                # Generate Google Maps Directions link
                google_maps_link = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={hospital.lat},{hospital.lon}&travelmode=driving"
                # Return as a dictionary suitable for JSON serialization and JavaScript use
                nearby.append({
                    'name': hospital.name,
                    'city': hospital.city,
                    'lat': hospital.lat,
                    'lon': hospital.lon,
                    'url': hospital.url,
                    'distance': f"{distance_in_km} km",
                    'travel_time': f"{travel_time_min} mins",
                    'maps_link': google_maps_link
                })
    return sorted(nearby, key=lambda x: float(x['distance'].split()[0]))





# --- PWA ROUTES ---
@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def serve_sw():
    return send_from_directory('static', 'sw.js')

# MIN_HYBRID_THRESHOLD = 5 # more lenient

@app.route("/")
def welcome():
    # Determine where the user should go after the 8-second animation
    if current_user.is_authenticated:
        next_url = url_for('home')
    else:
        next_url = url_for('login')

    return render_template("welcome.html", next_url=next_url)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    # If already logged in, no need to sign up
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        try:
            name = request.form.get("name")
            username = request.form.get("username")
            gender = request.form.get("gender")
            email = request.form.get("email")
            password = request.form.get("password")

            # Check if user already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('<p style="color:red">Username already exists. Please choose another</p>')
                return redirect(url_for("signup"))

            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('<p style="color:red">Email already exists. Please choose another.</p>')
                return redirect(url_for("signup"))

            # Hash the password
            hashed = bcrypt.generate_password_hash(password).decode('utf-8')

            # Create New User
            new_user = User(
                username=username,
                name=name,
                gender=gender,
                email=email,
                password=hashed
            )

            db.session.add(new_user)
            db.session.commit()

            # Optional: Log the activity
            activity = UserActivity(user_id=new_user.id, action="account_created")
            db.session.add(activity)
            db.session.commit()

            flash("Account created successfully! Please login.",)
            return redirect(url_for("login"))

        except Exception as e:
            db.session.rollback()
            print(f"Signup Error: {e}")
            flash("An error occurred. Please try again.")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)

            # Log Activity
            try:
                db.session.add(UserActivity(user_id=user.id, action="login"))
                db.session.commit()
            except Exception as e:
                print(f"Activity Log Error: {e}")

            return redirect(url_for("home"))
        else:
            flash('<p style ="color: red;">Invalid username or password</p>', 'warning')

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    db.session.add(UserActivity(user_id=current_user.id, action="logout"))
    db.session.commit()
    logout_user()
    return redirect(url_for("login"))


@app.route("/reset_request", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        user = User.query.filter_by(username=username, email=email).first()
        if user:
            token = get_reset_token(user.id)
            # In a real app, use Flask-Mail here.
            # For now, we print the link to your terminal to copy/paste.
            reset_link = url_for('reset_token', token=token, _external=True)
            print(f"\n[SECURITY] RESET LINK: {reset_link}\n")

            flash(f'A reset link has been generated! It expires in 30 mins. <a href="{reset_link}" style="color: #6ac36a; text-decoration: underline;">Click here to reset your password</a>', 'info')
            return redirect(url_for('login'))
        else:
            flash('<p style=" color:red">No account found with that username or email</p>', 'warning')
            return redirect(url_for('reset_request'))

    return render_template('reset_request.html')



@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    user = verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))

    if request.method == 'POST':
        password = request.form.get('password')
        # Use your existing bcrypt instance
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_token.html')


@app.route('/privacy_policy')
def privacy_policy():
    return render_template("privacy_policy.html")


@app.route('/terms_of_service')
def terms_of_service():
    return render_template('terms_of_service.html')


@app.route("/submit_vitals", methods=["POST"])
@login_required
def submit_vitals():
    # Use request.get_json() if sending via JavaScript fetch,
    # or request.form if sending via a standard HTML form.
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    try:
        # 1. Extract values
        temp = float(data.get('temp', 0))
        hr = int(data.get('hr', 0))
        bp_sys = int(data.get('bp_sys', 0))
        bp_dia = int(data.get('bp_dia', 0))
        spo2 = int(data.get('spo2', 0))

        # 2. Basic Logic for Severity & Advice
        severity = "Normal"
        advice_parts = []

        if temp > 38.0:
            severity = "Moderate"
            advice_parts.append("Fever detected.")
        elif temp < 35.0:
            severity = "High"
            advice_parts.append("Low body temperature (Hypothermia risk).")

        if spo2 < 94:
            severity = "High"
            advice_parts.append("Low oxygen levels detected.")

        if bp_sys >= 140 or bp_dia >= 90:
            severity = "Moderate"
            advice_parts.append("Blood pressure is elevated.")

        if not advice_parts:
            advice = "Your vitals appear to be within normal ranges."
        else:
            advice = " ".join(advice_parts) + " Please monitor and consult a professional if symptoms persist."

        # 3. Create and Save the Vitals Entry
        new_vitals = VitalsLog(
            user_id=current_user.id,
            temperature=temp,
            heart_rate=hr,
            bp_systolic=bp_sys,
            bp_diastolic=bp_dia,
            spo2=spo2,
            severity=severity,
            result=advice  # Storing the AI advice here
        )

        db.session.add(new_vitals)
        db.session.commit()

        return jsonify({
            "status": "success",
            "message": "Vitals saved successfully!",
            "severity": severity,
            "advice": advice
        }), 200

    except ValueError:
        return jsonify({"status": "error", "message": "Invalid input format. Please enter numbers."}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    # 1. Handle POST Request (AJAX Symptom Submission)
    if request.method == 'POST':
        data = request.get_json()
        user_input = data.get("symptoms", "").strip()

        # Get coordinates from frontend hidden inputs
        lat_str = data.get("lat")
        lon_str = data.get("lon")

        # --- EXTRACT QUESTIONNAIRE DATA ---
        severity = data.get("severity", "moderate").lower()
        recipient = data.get("patient", "self").lower()
        gender = data.get("gender", "male").lower()
        age_group = data.get("age", "adult").lower()
        last_diagnosed = data.get("last_diagnosed", "N/A")
        notice = data.get("notice_self") or data.get("notice_others") or "unknown"



        result = []
        hospitals_list = []
        audio_file = ""

        # Validate input
        if not user_input:
            return jsonify({"result": ["Please describe your symptoms."], "hospitals": []})

        # --- STEP 1: Grammar Engine ---
        if recipient == "self":
            subj, poss, v_has, v_is, v_needs = "You", "your", "have", "are", "need"
        else:
            subj = "He" if gender == "male" else "She"
            poss = "his" if gender == "male" else "her"
            v_has, v_is, v_needs = "has", "is", "needs"

        enhanced_input = f"{user_input}. Symptoms noticed {notice}."
        if recipient == "self" and last_diagnosed != "Never":
            enhanced_input += f" Patient has history of diagnosis: {last_diagnosed}."

        # --- STEP 2: Hybrid NLP/ML Logic ---
        ml_results = ml_predict_condition(user_input, top_n=5)
        spacy_matches = check_symptoms(user_input, min_score_threshold=1, top_n=5)

        hybrid_scores = {}
        for cond, prob in ml_results:
            hybrid_scores[cond] = {"ml": prob * 100, "nlp": 0}
        for cond, score in spacy_matches:
            if cond in hybrid_scores:
                hybrid_scores[cond]["nlp"] = score * 10
            else:
                hybrid_scores[cond] = {"ml": 0, "nlp": score * 10}

        ranked = sorted([(c, s["ml"] + s["nlp"]) for c, s in hybrid_scores.items()], key=lambda x: x[1], reverse=True)
        valid_conditions = [c for c, score in ranked if score >= 20][:1]

        # --- STEP 3: Generate Advice ---
        if valid_conditions:
            cond_key = valid_conditions[0].lower()
            cond_entry = symptom_data.get(cond_key)

            if cond_entry:
                advice_dict = cond_entry.get("advice", {})
                raw_text = advice_dict.get(severity, advice_dict.get("moderate", "Please consult a doctor."))
                final_msg = raw_text.format(subject=subj, possessive=poss, verb_has=v_has, verb_is=v_is,
                                            verb_needs=v_needs)

                if age_group == "minor":
                    final_msg = "🧒🏾 Minor: " + final_msg
                elif age_group == "elderly":
                    final_msg = "🧓🏾 Elderly: " + final_msg
                result.append(final_msg)
            else:
                result.append(
                    f"I recognized symptoms for {cond_key}, but I'm still learning the best advice for it. Please consult a doctor.")
        else:
            result.append(
                f"I couldn't match your symptoms precisely. Since you noticed this {notice}, please consult a doctor.")

        # Audio Generation
        audio_path = generate_audio(" ".join(result))
        audio_file = os.path.basename(audio_path)

        # --- STEP 4: HOSPITAL MAPPING (Improved Accuracy) ---
        if lat_str and lon_str:
            try:
                u_lat = float(lat_str)
                u_lon = float(lon_str)

                # Real-world adjustments for Nigeria:
                AVG_SPEED_KMH = 25  # Lower speed better reflects urban traffic/potholes
                ROAD_ADJUSTMENT = 1.3  # Adds 30% distance to account for road curves/turns

                all_db_hospitals = Hospital.query.all()


                for h in all_db_hospitals:
                    # 1. Straight-line calculation
                    dist_deg = ((u_lat - h.lat) ** 2 + (u_lon - h.lon) ** 2) ** 0.5
                    straight_line_km = dist_deg * 111

                    # 2. Road-distance calculation (The 'Google' distance)
                    dist_km = round(straight_line_km * ROAD_ADJUSTMENT, 1)

                    # Only process hospitals within a 50km road-distance radius
                    if dist_km <= 40:
                        h_is_emergency = False
                        if hasattr(h, 'severity_tag') and h.severity_tag == 'high':
                            h_is_emergency = True
                        elif "teaching" in h.name.lower() or "emergency" in h.name.lower():
                            h_is_emergency = True

                        # 3. Travel time calculation
                        # Calculate total hours first
                        total_hours = dist_km / AVG_SPEED_KMH

                        # Convert to total seconds (Hours * 3600)
                        total_seconds = int(total_hours * 3600) + 180  # Adding 180 seconds (3 mins) as your buffer

                        # Break down into H, M, S
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        seconds = total_seconds % 60

                        # Format the string based on whether hours exist
                        if hours > 0:
                            time_display = f"{hours}hr {minutes}min"
                        else:
                            time_display = f"{minutes}min"

                        # Google Maps link (Using the modern 'dir' API for better compatibility)
                        google_maps_link = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lon}&destination={h.lat},{h.lon}&travelmode=driving"

                        h_data = {
                            "name": h.name,
                            "city": h.city,
                            "lat": h.lat,
                            "lon": h.lon,
                            "url": h.url if h.url else "#",
                            "phone": getattr(h, 'phone_number', 'Not Available'),
                            "distance": f"{dist_km} km",
                            "travel_time": time_display,
                            "maps_link": google_maps_link
                        }

                        # Filter Logic
                        if severity == "high":
                            hospitals_list.append(h_data)
                        else:
                            if not h_is_emergency:
                                hospitals_list.append(h_data)

                # Sort by road distance
                hospitals_list.sort(key=lambda x: float(x['distance'].split()[0]))

            except Exception as e:
                print(f"Hospital Logic Error: {e}")

        # --- STEP 5: DATABASE SAVE ---
        try:
            new_report = SymptomReport(
                user_id=current_user.id,
                input_text=user_input,
                result=" ".join(result),
                location=f"{lat_str},{lon_str}",
                severity=severity,
                last_diagnosed=last_diagnosed,
                notice=notice,
                # Ensure these columns exist in your SymptomReport model:
                recipient=recipient,
                gender=gender,
                age=age_group,
                timestamp=datetime.utcnow()
            )
            db.session.add(new_report)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Database error: {e}")

        # 6. Final AJAX Response
        return jsonify({
            "result": result,
            "audio_file": url_for('static', filename=audio_file),
            "hospitals": hospitals_list,
            "severity": severity
        })

    # GET request: Show the chat page
    return render_template('chat.html')


@app.route("/home", methods=["GET", "POST"])
@login_required
def home():
    # 1. Handle Feedback Submission (POST) remains the same
    if request.method == "POST":
        feedback_content = request.form.get("feedback_message")
        if feedback_content:
            try:
                new_feedback = Feedback(
                    message=feedback_content,
                    user_id=current_user.id,
                    timestamp=datetime.utcnow()
                )
                db.session.add(new_feedback)
                db.session.commit()
                return jsonify({"status": "success", "message": "Thank you!"}), 200
            except Exception as e:
                db.session.rollback()
                return jsonify({"status": "error", "message": str(e)}), 500

    # 2. Handle Page Load (GET)
    # Fetch Symptom History
    # 1. Fetch both histories
    symptoms = SymptomReport.query.filter_by(user_id=current_user.id).all()
    # Ensure you have a VitalsLog model defined!
    vitals = VitalsLog.query.filter_by(user_id=current_user.id).all()

    # Sort combined list by timestamp
    all_history = sorted(symptoms + vitals, key=lambda x: x.timestamp, reverse=True)

    # Fetch Hospitals
    all_hospitals = Hospital.query.all()
    hospitals = [{
        "name": h.name,
        "city": h.city,
        "lat": h.lat,
        "lon": h.lon,
        "url": h.url or "#",
        "phone": getattr(h, 'phone_number', 'N/A')
    } for h in all_hospitals]

    return render_template('home.html',
                           hospitals=hospitals,
                           all_history=all_history,
                           current_user=current_user)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
