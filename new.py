from flask import Flask, request, render_template_string, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import spacy

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Configure SQLite
db_path = 'sqlite:///symptoms.db'
app.config['SQLALCHEMY_DATABASE_URI'] = db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# DB Models
class SymptomReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String(500))
    location = db.Column(db.String(100))
    result = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(1000))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Hospital(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    lat = db.Column(db.Float, nullable=True)
    lon = db.Column(db.Float, nullable=True)

# Create DB tables
with app.app_context():
    db.create_all()
    # Ensure sample hospitals are only added if the table is empty
    if Hospital.query.count() == 0:
        sample_hospitals = [
            Hospital(name="General Hospital Lagos", city="lagos", lat=6.5244, lon=3.3792),
            Hospital(name="University Teaching Hospital", city="port harcourt", lat=4.8156, lon=7.0498),
            Hospital(name="Abuja Specialist Hospital", city="abuja", lat=9.0579, lon=7.4951),
            # Hospitals near Delta State (as previously added for your location)
            Hospital(name="Central Hospital Warri", city="warri", lat=5.5222, lon=5.7533),
            Hospital(name="Delta State University Teaching Hospital (DELSUTH)", city="omere", lat=5.5000, lon=6.0000), # Oghara/Omeru is where DELSUTH is
            Hospital(name="Federal Medical Centre Asaba", city="asaba", lat=6.1959, lon=6.7323),
            Hospital(name="University of Benin Teaching Hospital (UBTH)", city="benin city", lat=6.3475, lon=5.5866),
            Hospital(name="Federal Medical Centre Yenagoa", city="yenagoa", lat=5.2017, lon=6.2625),
            Hospital(name="Gracepoint Hospital Warri", city="warri", lat=5.5100, lon=5.7700),
            Hospital(name="Lily Hospitals Ltd. Warri", city="warri", lat=5.5150, lon=5.7650),
            Hospital(name="Precious Hospital Asaba", city="asaba", lat=6.1800, lon=6.7000)
        ]
        db.session.bulk_save_objects(sample_hospitals)
        db.session.commit()

symptom_data = {
    "malaria": {
        "symptoms": ["fever", "chills", "sweating", "headache"],
        "advice": "You may have malaria. Please visit a health center."
    },
    "typhoid": {
        "symptoms": ["fever", "stomach", "diarrhea", "weakness"],
        "advice": "Possible typhoid. Seek medical attention."
    },
    "cold": {
        "symptoms": ["cough", "sneezing", "runny", "nose", "sore", "throat"],
        "advice": "Looks like common cold. Stay hydrated and rest."
    }
}

def check_symptoms(user_input):
    doc = nlp(user_input.lower())
    tokens = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']]
    matched_conditions = []
    for condition, details in symptom_data.items():
        for symptom in details["symptoms"]:
            if symptom in tokens:
                matched_conditions.append(condition)
                break
    return list(set(matched_conditions))

def find_nearby_hospitals(lat, lon, radius_km=100):
    nearby = []
    for hospital in Hospital.query.all():
        if hospital.lat and hospital.lon:
            # This is a very simplified distance calculation (Euclidean on lat/lon).
            # For real-world applications, consider Haversine formula or a spatial database.
            distance = ((lat - hospital.lat)**2 + (lon - hospital.lon)**2)**0.5
            if distance <= (radius_km / 111):  # Roughly 111 km per degree
                nearby.append(hospital)
    return nearby

main_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Health Advisor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        h2 { color: #0056b3; }
        form { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        label { display: block; margin-bottom: 8px; font-weight: bold; }
        input[type="text"] { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #28a745; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #218838; }
        h3, h4 { color: #0056b3; margin-top: 20px; }
        ul { list-style: none; padding: 0; }
        li { background-color: #e9ecef; padding: 10px; margin-bottom: 5px; border-radius: 4px; }
        p { margin-top: 15px; color: #666; }
    </style>
    <script>
        function getLocation() {
            console.log("Attempting to get location...");
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(showPosition, showError);
            } else {
                console.log("Geolocation is not supported by this browser.");
                alert("Geolocation is not supported by this browser. Cannot provide nearby hospitals.");
            }
        }

        function showPosition(position) {
            console.log("Location obtained successfully!");
            document.getElementById("lat").value = position.coords.latitude;
            document.getElementById("lon").value = position.coords.longitude;
            console.log("Lat:", position.coords.latitude, "Lon:", position.coords.longitude);
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
            alert("Could not get your location: " + errorMessage + " Nearby hospitals cannot be found.");
        }

        window.onload = getLocation;
    </script>
</head>
<body>
    <h2>AI Health Advisor</h2>
    <form method="POST">
        <label for="symptoms">Enter your symptoms:</label><br>
        <input type="text" id="symptoms" name="symptoms" required placeholder="e.g., fever, headache, body aches"><br>
        <input type="hidden" name="lat" id="lat">
        <input type="hidden" name="lon" id="lon">
        <br><button type="submit">Get Advice & Find Hospitals</button>
    </form>
    {% if result %}
        <hr>
        <h3>Advice:</h3>
        <ul>
        {% for item in result %}<li>{{ item }}</li>{% endfor %}
        </ul>
        {% if hospitals %}
            <h4>Nearby Hospitals:</h4>
            <ul>
            {% for hospital in hospitals %}
                <li>{{ hospital.name }} ({{ hospital.city.title() }})</li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No hospitals found nearby based on your current location or if location access was denied/unavailable.</p>
        {% endif %}
    {% endif %}
</body>
</html>
"""

# Flask Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    result = None       # Initialize result variable
    hospitals = []      # Initialize hospitals list
    user_location = ""  # Initialize user_location (though not directly used in template, good for context)

    if request.method == "POST":
        user_input = request.form["symptoms"] # Get symptoms from the form
        lat = request.form.get("lat")         # Get latitude from the hidden input
        lon = request.form.get("lon")         # Get longitude from the hidden input

        try:
            # Convert lat/lon to float if they exist
            lat = float(lat)
            lon = float(lon)
            hospitals = find_nearby_hospitals(lat, lon) # Find nearby hospitals
        except (ValueError, TypeError): # Handle cases where lat/lon might be missing or invalid
            print("Geolocation not available or invalid.")
            pass # Continue without nearby hospitals if geolocation fails

        conditions = check_symptoms(user_input) # Check symptoms against known data
        if conditions:
            # If conditions are found, format the advice
            result = [f"{cond.upper()}: {symptom_data[cond]['advice']}" for cond in conditions]
        else:
            # If no conditions match, provide general advice
            result = ["No known match. Please consult a doctor."]

        # Store the symptom report in the database
        db_result = "; ".join(result) # Join multiple advice strings for storage
        report = SymptomReport(input_text=user_input, location=f"{lat},{lon}" if lat and lon else "N/A", result=db_result)
        db.session.add(report)
        db.session.commit()

    # Render the HTML template with the results and hospitals
    return render_template_string(main_template, result=result, hospitals=hospitals)

# Run the Flask application in debug mode (useful for development)
if __name__ == "__main__":
    app.run(debug=True)