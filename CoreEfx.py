from flask import Flask, request, redirect, render_template_string, render_template, jsonify, url_for, flash
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
    input_text = db.Column(db.Text)  # The raw text input from the user (symptoms)
    location = db.Column(db.String(100))  # User's approximate location (latitude,longitude string)
    result = db.Column(db.Text)  # The health advice/diagnosis provided by the system (increased length)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # Time when the report was created
    severity = db.Column(db.String(20))  # Low, Moderate, High
    recipient = db.Column(db.String(50))  # "Myself" or "Someone else"
    age = db.Column(db.String(50))
    gender = db.Column(db.String(20))
    audio_url = db.Column(db.String(255))
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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
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
            "low": "\n{subject} {verb_has} signs of mild malaria. {subject} should rest and get a blood test soon.",
            "moderate": "\nYou likely {verb_has} malaria. Please visit the health center quickly for a blood test and ACT treatment.",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} showing severe malaria symptoms. Go to the hospital immediately for a drip."
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
            "low": "\n{subject} should rest and drink only safe, boiled water. Monitor the fever closely.",
            "moderate": "\nThis may be typhoid fever. {subject} {verb_needs} to visit the nearest health center for a test and antibiotics.",
            "high": "\n🚨 CRITICAL: {subject} {verb_has} severe typhoid symptoms. Go to the hospital immediately to check for internal complications."
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
            "low": "\nThis looks like a common cold. {subject} should rest and drink plenty of fluids.",
            "moderate": "\n{subject} {verb_has} a persistent cold. If {possessive} throat pain gets worse, consider seeing a health worker.",
            "high": "\n⚠️ WARNING: If {subject} {verb_is} having real trouble breathing or a very high fever with this cold, go to the clinic now."
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
            "low": "\nThis could be the flu. {subject} {verb_needs} plenty of bed rest and warm fluids.",
            "moderate": "\n{subject} {verb_has} moderate flu symptoms. Monitor {possessive} breathing and visit a health worker if it does not improve in 3 days.",
            "high": "\n🚨 URGENT: {subject} {verb_has} severe flu signs. Go to the hospital immediately if {subject} {verb_is} gasping for air."
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
            "low": "\n{subject} {verb_has} mild diarrhea. Start drinking ORS (salt and sugar solution) immediately to stay hydrated.",
            "moderate": "\nThis may be a diarrheal infection. {subject} {verb_needs} to see a health worker if {subject} cannot stop vomiting.",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} losing too much water. Go to the hospital for a drip immediately."
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
            "low": "\nThis looks like a urinary tract infection. {subject} should drink plenty of clean water to wash {possessive} system and not hold pee.",
            "moderate": "\n{subject} {verb_needs} to visit a health center for a test and proper antibiotics for this infection.",
            "high": "\n🚨 EMERGENCY: If {subject} {verb_has} high fever or back pain with these symptoms, seek medical care immediately as it may have reached the kidneys."
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
            "low": "\nThis may be a skin infection. {subject} should avoid scratching and keep the area clean and dry.",
            "moderate": "\n{subject} should see a doctor for a medicated cream or medicine, especially if the rash is spreading.",
            "high": "\n🚨 URGENT: If the skin is peeling, very painful, or {subject} {verb_has} a very high fever, visit the hospital now."
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
            "low": "\n{subject} may be dehydrated. {subject} should go to a cool place and drink clean water or ORS.",
            "moderate": "\n{subject} {verb_is} showing signs of heat exhaustion. Rest with legs raised and keep drinking fluids.",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} severely dehydrated. If {subject} cannot drink, go to the hospital for a drip immediately."
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
            "low": "\nThis may be due to stress or fatigue. {subject} should rest in a dark room and stay hydrated.",
            "moderate": "\n{subject} can take paracetamol. If the pain continues, {subject} should be checked for malaria or high blood pressure.",
            "high": "\n🚨 URGENT: If this is the worst headache {subject} {verb_has} ever felt, or if {possessive} neck is stiff, go to the hospital immediately."
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
            "low": "\n{subject} {verb_has} general body aches. This may be due to fatigue or minor stress. {subject} should rest and try a warm bath.",
            "moderate": "\n{subject} {verb_has} significant body pain. {subject} can take paracetamol and stay hydrated, but visit a clinic if it persists.",
            "high": "\n🚨 NOTICE: {subject} {verb_is} in severe pain. This could be a sign of a serious infection like malaria or meningitis. Please see a doctor immediately."
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
            "low": "\nThese could be early pregnancy signs. {subject} should take a test to confirm the condition.",
            "moderate": "\nIf confirmed, {subject} should visit a health center to start antenatal care and take pregnancy vitamins.",
            "high": "\n🚨 URGENT: If {subject} {verb_has} severe lower belly pain or heavy bleeding, go to the hospital immediately."
        }
    },

    "eye infection (Apollo/conjunctivitis)": {
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
            "low": "\nThis may be Apollo. {subject} should avoid touching {possessive} eyes and wash {possessive} hands frequently.",
            "moderate": "\n{subject} {verb_needs} to visit a health center for proper antibiotic eye drops.",
            "high": "\n🚨 URGENT: If {subject} cannot see clearly or has severe pain, {subject} must see an eye doctor immediately."
        }
    },

    "blurred vision/eye problem": {
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
            "low": "\n{possessive} vision is slightly blurry. {subject} should rest {possessive} eyes and avoid bright screens for now.",
            "moderate": "\n{subject} {verb_has} noticeable eye trouble. {subject} {verb_needs} an urgent check-up with an eye specialist (ophthalmologist).",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} experiencing rapid vision loss or intense eye pain. This is a medical emergency. Go to an eye clinic right now."
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
            "low": "\n{subject} {verb_has} a chest infection. Monitor {possessive} breathing closely and stay warm.",
            "moderate": "\nThis could be pneumonia. {subject} {verb_needs} a medical exam and likely antibiotics from a health center.",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} struggling to breathe. Seek oxygen and emergency medical care immediately."
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
            "low": "\n{subject} {verb_has} a persistent cough. Because it has lasted long, {subject} should go for a free TB test at a health center.",
            "moderate": "\nThis could be tuberculosis. It can spread to others in the house. {subject} must visit a health worker for testing and free treatment immediately.",
            "high": "\n🚨 CRITICAL: {subject} {verb_is} coughing blood and losing weight rapidly. {subject} {verb_needs} immediate hospital admission for TB care."
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
            "low": "\nThis may be diabetes. {subject} should reduce sugar intake and go for a blood sugar test.",
            "moderate": "\n{subject} {verb_needs} to see a doctor to manage {possessive} sugar levels and get proper medication.",
            "high": "\n🚨 EMERGENCY: If {subject} {verb_is} confused, vomiting, or very weak, {possessive} sugar may be dangerously high. Go to the hospital now."
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
            "low": "\n{possessive} blood pressure may be slightly high. {subject} should rest, reduce salt, and check the BP again later.",
            "moderate": "\nThis looks like hypertension. {subject} {verb_needs} to see a doctor for a proper BP check and lifestyle advice.",
            "high": "\n🚨 DANGER: {possessive} blood pressure is very high. This can lead to a stroke. Go to the hospital immediately."
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
            "low": "\n{subject} should avoid stress and salty foods. Monitor if the chest tightness continues.",
            "moderate": "\nThis may be a heart problem. {subject} {verb_needs} a check-up with a cardiologist for an ECG or scan.",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} showing signs of a heart attack. Go to the emergency room right now."
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
            "low": "\n{possessive} symptoms may be due to mild malnutrition. Please provide balanced meals with proteins (beans, eggs, fish) and vitamins.",
            "moderate": "\n{subject} {verb_has} signs of moderate malnutrition. {subject} {verb_needs} a nutrition plan and supplements from a health center.",
            "high": "\n🚨 CRITICAL: This is severe malnutrition (possible Kwashiorkor or Marasmus). {subject} must be taken to a stabilization center or hospital immediately for therapeutic feeding."
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
            "low": "\n{subject} should avoid spicy foods and soda. Do not take pain killers like Ibuprofen on an empty stomach.",
            "moderate": "\nThis could be an ulcer. {subject} {verb_needs} to see a doctor for treatment to protect the stomach lining.",
            "high": "\n🚨 CRITICAL: {subject} {verb_is} showing signs of internal bleeding (black stool/vomiting blood). See a doctor immediately."
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
            "low": "\n{subject} should stay away from dust and smoke. Keep the inhaler close.",
            "moderate": "\n{possessive} asthma seems to be acting up. {subject} {verb_needs} to use a preventer inhaler and see a doctor.",
            "high": "\n🚨 EMERGENCY: {subject} cannot breathe well. Use the rescue inhaler and go to the hospital immediately for oxygen."
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
            "low": "\nThis looks like mild tonsillitis. {subject} should drink warm water and gargle with salt water to soothe the pain.",
            "moderate": "\n{subject} {verb_has} moderate tonsillitis. {subject} likely {verb_needs} antibiotics. Please visit a health worker if the fever stays high.",
            "high": "\n🚨 URGENT: If {subject} {verb_is} having a very hard time swallowing or breathing because of the swelling, go to the emergency room immediately."
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
            "low": "\n{subject} {verb_has} signs of liver irritation. Rest well and avoid all alcohol and herbal mixtures.",
            "moderate": "\nThis may be Hepatitis. {subject} {verb_needs} a blood test (HBsAg) at the hospital to know the type.",
            "high": "\n🚨 URGENT: {subject} {verb_is} very ill with jaundice. Go to the hospital for liver support treatment."
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
            "low": "\n{subject} should drink plenty of water to help flush the system.",
            "moderate": "\nThis could be kidney stones. {subject} {verb_needs} an ultrasound scan and proper pain medication.",
            "high": "\n🚨 URGENT: {subject} {verb_is} in extreme pain or cannot pass urine. Go to the hospital immediately."
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
            "low": "\n{subject} should eat more iron-rich foods like green vegetables and liver.",
            "moderate": "\n{possessive} blood level (PCV) might be low. {subject} {verb_needs} a blood test and iron supplements.",
            "high": "\n🚨 CRITICAL: {subject} {verb_is} severely anemic. {subject} may need a blood transfusion. Go to the hospital now."
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
            "low": "\nEnsure {subject} gets enough sleep and avoids triggers. Monitor any small 'absent' moments.",
            "moderate": "\n{subject} {verb_needs} to see a neurologist to start daily medication to prevent future fits.",
            "high": "\n🚨 EMERGENCY: The seizure has lasted too long or {subject} is not waking up. Seek emergency care immediately."
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
            "low": "\n{subject} {verb_has} mild menstrual pain. {subject} should use a hot water bottle on {possessive} belly and rest.",
            "moderate": "\nThis looks like dysmenorrhea. {subject} {verb_needs} pain medicine like paracetamol. If the pain no gree go, see a doctor.",
            "high": "\n⚠️ {subject} {verb_is} in severe pain. If {subject} {verb_is} fainting or bleeding too much, please go to the hospital immediately."
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
            "low": "\n⚠️ Note: {subject} must monitor {possessive} blood pressure daily. Any increase is a danger sign.",
            "moderate": "\n{subject} {verb_has} signs of preeclampsia. {subject} {verb_needs} to see {possessive} doctor today for a check-up.",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} showing critical signs of preeclampsia. Rush to the hospital now; this is very serious for both {subject} and the baby."
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
            "low": "\nThis could be a minor bone injury. {subject} should rest and keep the limb still.",
            "moderate": "\nThis could be a fractured or broken bone. {subject} should use a stick or hard material to keep the place straight and go to the hospital.",
            "high": "\n🚨 EMERGENCY: This is a severe broken bone. Do not move {subject}. Keep the limb completely still and call for an ambulance or go to the emergency room immediately."
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
            "low": "\n{subject} {verb_has} an allergy. {subject} should take an antihistamine and stay away from the cause.",
            "moderate": "\n{possessive} allergy is worsening. {subject} {verb_needs} medical attention before the throat closes.",
            "high": "\n🚨 LIFE-THREATENING: {subject} {verb_is} in anaphylactic shock. {subject} cannot breathe well. Rush to the hospital for an adrenaline (EpiPen) injection now!"
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
            "low": "\n{subject} {verb_has} signs of Bilharzia. {subject} should avoid swimming in fresh water (rivers/ponds) and get a urine test.",
            "moderate": "\nThis could be Schistosomiasis. {subject} {verb_needs} proper deworming medicine (Praziquantel) from a health center.",
            "high": "\n🚨 WARNING: Heavy blood in the urine or stool indicates a severe infection. {subject} must visit the hospital for a full check-up."
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
            "low": "\n{subject} may have gallstones. {subject} should avoid oily and fatty foods to prevent the pain from returning.",
            "moderate": "\nThis looks like a gallbladder issue. {subject} {verb_needs} an abdominal scan at a clinic to confirm if stones are present.",
            "high": "\n🚨 EMERGENCY: If {subject} {verb_is} vomiting and {possessive} eyes are yellow (jaundice), the stone might be blocking a duct. Go to the hospital now."
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
            "low": "\n{subject} likely {verb_has} Dengue. {subject} should rest and take only Paracetamol. Do NOT take Ibuprofen or Aspirin.",
            "moderate": "\nThis looks like Dengue fever. {subject} {verb_needs} to stay very hydrated and be monitored by a health worker for any bleeding.",
            "high": "\n🚨 CRITICAL: If {subject} starts bleeding from the gums or nose, it is Dengue Hemorrhagic Fever. Seek emergency care immediately."
        }
    },

    "stroke": {
        "symptoms": [
            "one side of face drop",
            "one hand or leg no fit move",
            "trouble talk",
            "sudden onset of weakness",
            "facial drooping",
            "slurred speech",
            "loss of balance",
            "severe headache"
        ],
        "advice": {
            "low": "\n⚠️ Even if symptoms are mild, {subject} {verb_needs} a brain scan immediately to prevent a full stroke.",
            "moderate": "\n{subject} {verb_is} showing signs of a stroke. Do not wait. Take {subject} to the hospital right now.",
            "high": "\n🚨 CRITICAL EMERGENCY: {subject} {verb_is} having a major stroke. Every minute counts to save {possessive} brain. Rush to the nearest Emergency Center immediately!"
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
            "low": "\n{subject} should monitor the pain. If it moves to the lower right side, it could be the appendix.",
            "moderate": "\nThis may be appendicitis. {subject} should not eat or drink anything and go to the hospital for a scan.",
            "high": "\n🚨 URGENT: The appendix may burst. {subject} {verb_needs} surgery immediately. Go to the hospital emergency room now."
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
            "low": "\n{subject} {verb_has} a persistent fever. Monitor {possessive} symptoms very closely for any bleeding.",
            "moderate": "\nThis could be Lassa Fever. {subject} {verb_needs} to go to the nearest health center immediately for isolation and treatment.",
            "high": "\n🚨 CRITICAL: This is a life-threatening infection. {subject} {verb_is} showing severe Lassa Fever signs. Seek specialized medical help immediately."
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
            "low": "\n{subject} may have hemorrhoids or piles. {subject} should eat high-fiber food and drink lots of water.",
            "moderate": "\nFor these piles, {subject} should avoid straining when {subject} {verb_is} stooling. See a doctor for ointment.",
            "high": "\n⚠️ NOTICE: If the pain is severe or the bleeding is heavy, {subject} {verb_needs} to see a doctor for possible surgery or advanced care."
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
            "low": "\n{subject} may have an infection. {subject} should get a test at a clinic soon.",
            "moderate": "\nThis looks like Gonorrhea (STD). {subject} {verb_needs} a doctor for a test and antibiotics. Do not have sex until treatment is complete.",
            "high": "\n⚠️ URGENT: The infection may be spreading. {subject} must see a doctor immediately to prevent long-term damage like PID."
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
            "low": "\n{subject} {verb_has} symptoms that could be an early STD like Syphilis. {subject} should avoid sexual contact and get a blood test soon.",
            "moderate": "\nThis looks like Syphilis. {subject} {verb_needs} a penicillin injection from a clinic to stop the infection from spreading.",
            "high": "\n🚨 URGENT: Syphilis can affect the brain and heart if left too long. Since {subject} {verb_is} showing advanced signs, see a specialist today."
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
            "low": "\nIt could be that {possessive} sugar level is low. {subject} should quickly eat something sweet like sugar or juice.",
            "moderate": "\nThis is hypoglycemia. {subject} {verb_needs} to eat a proper meal now and check {possessive} sugar level.",
            "high": "\n🚨 EMERGENCY: {subject} {verb_is} at risk of fainting or a coma. Give {subject} sugar immediately and go to the hospital if {subject} does not wake up fully."
        }
    },
    "glaucoma (eye pressure)": {
        "symptoms": [
            "eye pain strong",
            "headache bad",
            "eye no see clear fast",
            "eye red",
            "severe eye pain",
            "sudden blurred vision",
            "seeing halos",
            "redness of eye"
        ],
        "advice": {
            "low": "\n{subject} should have {possessive} eye pressure checked by an eye doctor soon.",
            "moderate": "\nThis may be Glaucoma. High pressure can blind {subject} fast. {subject} {verb_needs} to see an ophthalmologist today.",
            "high": "\n🚨 EMERGENCY: Sudden vision loss or severe eye pain is a medical emergency. Go to an eye clinic right now."
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
    "soreness", "sometimes no symptoms at all", "spit blood", "sticky discharge (eye gum when you wake)",
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
                flash("Username already exists. Please choose another.")
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

            flash("Account created successfully! Please login.")
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
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


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
            if cond_key in symptom_data:
                cond_entry = symptom_data[cond_key]
                raw_text = cond_entry.get("advice", {}).get(severity, cond_entry.get("advice", {}).get("moderate",
                                                                                                       "Please consult a doctor."))

                final_msg = raw_text.format(subject=subj, possessive=poss, verb_has=v_has, verb_is=v_is,
                                            verb_needs=v_needs)

                if age_group == "child":
                    final_msg = "👶 PEDIATRIC: " + final_msg
                elif age_group == "elderly":
                    final_msg = "👴 SENIOR: " + final_msg
                result.append(final_msg)
        else:
            result.append(
                f"I couldn't match your symptoms precisely. Given your {severity} severity, please see a doctor.")

        # Audio Generation
        audio_path = generate_audio(" ".join(result))
        audio_file = os.path.basename(audio_path)

        # --- STEP 4: HOSPITAL MAPPING (With Severity Filtering) ---
        if lat_str and lon_str:
            try:
                u_lat = float(lat_str)
                u_lon = float(lon_str)

                all_db_hospitals = Hospital.query.all()

                for h in all_db_hospitals:
                    # Search radius: approx 30km
                    if abs(h.lat - u_lat) < 0.3 and abs(h.lon - u_lon) < 0.3:

                        # Identify if the hospital is an emergency facility
                        # We check a 'severity_tag' column or if 'Teaching'/'Emergency' is in the name
                        h_is_emergency = False
                        if hasattr(h, 'severity_tag') and h.severity_tag == 'high':
                            h_is_emergency = True
                        elif "teaching" in h.name.lower() or "emergency" in h.name.lower():
                            h_is_emergency = True

                        h_data = {
                            "name": h.name,
                            "city": h.city,
                            "lat": h.lat,
                            "lon": h.lon,
                            "url": h.url if h.url else "#",
                            "phone": getattr(h, 'phone_number', 'Not Available')
                        }

                        # --- FILTER LOGIC ---
                        if severity == "high":
                            # If user is in High severity, show EVERYTHING (Emergency + General)
                            hospitals_list.append(h_data)
                        else:
                            # If severity is Low or Moderate, ONLY show non-emergency clinics
                            if not h_is_emergency:
                                hospitals_list.append(h_data)

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
    # Initialize variables for the template
    result = []
    audio_file = ""
    user_input = ""
    feedback_message = ""

    # 1. Fetch History Reports for the user
    history_reports = SymptomReport.query.filter_by(user_id=current_user.id).order_by(
        SymptomReport.timestamp.desc()
    ).all()

    # 2. Fetch all hospitals for the initial map display
    all_hospitals = Hospital.query.all()
    hospitals = [
        {
            "name": h.name,
            "city": h.city,
            "lat": h.lat,
            "lon": h.lon,
            "url": h.url if h.url else "#",
            "phone": getattr(h, 'phone_number', 'N/A')
        }
        for h in all_hospitals
    ]

    # 3. Handle Feedback Submission (Form Post)
    if request.method == "POST":
        feedback_content = request.form.get("feedback_message")
        if feedback_content:
            try:
                new_feedback = Feedback(
                    message=feedback_content,
                    user_id=current_user.id if current_user.is_authenticated else None,
                    timestamp=datetime.utcnow()
                )
                db.session.add(new_feedback)
                db.session.commit()
                flash("Thank you for your feedback!", "success")
            except Exception as e:
                db.session.rollback()
                print(f"Feedback Error: {e}")
        else:
            flash("Feedback message cannot be empty.", "warning")

    # 4. Render the external home.html file
    return render_template('home.html',
                           result=result,
                           hospitals=hospitals,
                           history_reports=history_reports,
                           feedback_message=feedback_message,
                           audio_file=audio_file,
                           current_user=current_user,
                           user_message=user_input)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
