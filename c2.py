# This is a complete, single-file Flask application for a service provider directory.
# It includes the backend API using Flask and a simple HTML/JavaScript front-end.

# To run this, you need to install Flask and firebase-admin:
# pip install Flask firebase-admin

# You also need to set up Firebase credentials.
# Download your service account key JSON file from the Firebase console and
# update the 'cred = credentials.Certificate(...)' line below with the path to your file.

from flask import Flask, request, jsonify, render_template_string
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os

# --- Firebase Initialization ---
# The environment variable '__firebase_config' is provided by the canvas environment.
# We'll use it to initialize the app.
try:
    # Attempt to get the Firebase config from the environment first
    firebase_config = json.loads(os.environ.get('__firebase_config', '{}'))
    if firebase_config:
        cred = credentials.Certificate(firebase_config)
    else:
        # If running locally, you must provide the path to your service account key file.
        # Download this from Firebase Project Settings > Service accounts > Generate new private key.
        # Place the JSON file in the same directory as this script.
        cred = credentials.Certificate("service-account-key.json")

    # Initialize the Firebase app with the credentials
    firebase_admin.initialize_app(cred)

except ValueError:
    # This handles cases where the app is already initialized, for example,
    # in some cloud environments, preventing a re-initialization error.
    pass
except FileNotFoundError:
    print(
        "Error: 'service-account-key.json' not found. Please download your Firebase service account key and place it in the project directory.")
    exit(1)

db = firestore.client()

# The '__app_id' is also provided by the canvas environment.
app_id = os.environ.get('__app_id', 'default-app-id')
# In a real app, the user ID would come from an authentication system.
user_id = "example-user-id-123"

app = Flask(__name__)


# --- Firestore Functions ---

def get_providers_from_db():
    """Retrieves all service providers from the database."""
    print(f"Retrieving providers for app_id: {app_id}")
    collection_path = f"artifacts/{app_id}/public/data/providers"
    providers_ref = db.collection(collection_path)
    providers_list = []
    try:
        docs = providers_ref.stream()
        for doc in docs:
            provider_data = doc.to_dict()
            provider_data['id'] = doc.id
            providers_list.append(provider_data)
        print(f"Found {len(providers_list)} providers.")
        return providers_list
    except Exception as e:
        print(f"Error getting providers: {e}")
        return []


def add_provider_to_db(provider_data):
    """Adds a new service provider to the database."""
    print(f"Adding new provider for app_id: {app_id}")
    collection_path = f"artifacts/{app_id}/public/data/providers"
    providers_ref = db.collection(collection_path)

    data_to_add = {
        **provider_data,
        "ownerId": user_id,
        "createdAt": firestore.SERVER_TIMESTAMP
    }

    try:
        doc_ref = providers_ref.add(data_to_add)
        doc_id = doc_ref[1].id
        print(f"Provider added with ID: {doc_id}")
        return doc_id
    except Exception as e:
        print(f"Error adding provider: {e}")
        return None


# --- Flask Routes (API Endpoints) ---

@app.route('/')
def index():
    """Serves the main HTML page for the application."""
    # The HTML is embedded directly into the Python file for simplicity.
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Service Provider Directory</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
            body {
                font-family: 'Inter', sans-serif;
            }
        </style>
    </head>
    <body class="bg-gray-100 flex flex-col items-center p-8 min-h-screen">

        <main class="w-full max-w-4xl">
            <h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Service Provider Directory</h1>

            <!-- Form to Add a New Provider -->
            <div class="bg-white rounded-xl shadow-lg p-6 mb-8">
                <h2 class="text-2xl font-semibold mb-4 text-gray-700">Add a New Provider</h2>
                <div id="status-message" class="text-center p-2 rounded-md transition-opacity duration-300 opacity-0"></div>
                <form id="provider-form" class="space-y-4">
                    <div>
                        <label for="name" class="block text-sm font-medium text-gray-700">Provider Name</label>
                        <input type="text" id="name" name="name" required
                               class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 border">
                    </div>
                    <div>
                        <label for="serviceType" class="block text-sm font-medium text-gray-700">Service Type</label>
                        <input type="text" id="serviceType" name="serviceType" required
                               class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 border">
                    </div>
                    <div>
                        <label for="description" class="block text-sm font-medium text-gray-700">Description</label>
                        <textarea id="description" name="description" rows="3" required
                                  class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 border"></textarea>
                    </div>
                    <div>
                        <label for="contactInfo" class="block text-sm font-medium text-gray-700">Contact Info</label>
                        <input type="text" id="contactInfo" name="contactInfo" required
                               class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2 border">
                    </div>
                    <button type="submit"
                            class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out">
                        Add Provider
                    </button>
                </form>
            </div>

            <!-- List of Providers -->
            <div class="bg-white rounded-xl shadow-lg p-6">
                <h2 class="text-2xl font-semibold mb-4 text-gray-700">Existing Providers</h2>
                <div id="providers-list" class="space-y-4">
                    <!-- Providers will be rendered here by JavaScript -->
                    <div class="text-center text-gray-500">Loading providers...</div>
                </div>
            </div>
        </main>

        <script>
            document.addEventListener('DOMContentLoaded', () => {
                const providersList = document.getElementById('providers-list');
                const providerForm = document.getElementById('provider-form');
                const statusMessage = document.getElementById('status-message');

                // Function to show a temporary message
                const showMessage = (message, type) => {
                    statusMessage.textContent = message;
                    statusMessage.classList.remove('opacity-0');
                    if (type === 'success') {
                        statusMessage.className = 'text-green-700 bg-green-100 text-center p-2 rounded-md transition-opacity duration-300';
                    } else if (type === 'error') {
                        statusMessage.className = 'text-red-700 bg-red-100 text-center p-2 rounded-md transition-opacity duration-300';
                    }
                    setTimeout(() => {
                        statusMessage.classList.add('opacity-0');
                    }, 5000); // Hide the message after 5 seconds
                };

                // Function to fetch and render providers
                const fetchProviders = async () => {
                    providersList.innerHTML = '<div class="text-center text-gray-500">Loading providers...</div>';
                    try {
                        const response = await fetch('/api/providers');
                        if (!response.ok) {
                            throw new Error('Failed to fetch providers');
                        }
                        const providers = await response.json();
                        renderProviders(providers);
                    } catch (error) {
                        console.error('Error fetching providers:', error);
                        providersList.innerHTML = `<div class="text-center text-red-500">Error: ${error.message}</div>`;
                    }
                };

                // Function to render providers to the DOM
                const renderProviders = (providers) => {
                    providersList.innerHTML = '';
                    if (providers.length === 0) {
                        providersList.innerHTML = '<div class="text-center text-gray-500">No providers found.</div>';
                        return;
                    }

                    providers.forEach(provider => {
                        const providerCard = document.createElement('div');
                        providerCard.className = 'bg-gray-50 rounded-lg p-4 border border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200';
                        providerCard.innerHTML = `
                            <h3 class="text-xl font-medium text-gray-800">${provider.name}</h3>
                            <p class="text-sm font-semibold text-indigo-600">${provider.serviceType}</p>
                            <p class="mt-2 text-gray-600">${provider.description}</p>
                            <p class="mt-2 text-sm text-gray-500">Contact: ${provider.contactInfo}</p>
                        `;
                        providersList.appendChild(providerCard);
                    });
                };

                // Event listener for form submission
                providerForm.addEventListener('submit', async (e) => {
                    e.preventDefault();

                    const formData = new FormData(providerForm);
                    const providerData = Object.fromEntries(formData.entries());

                    try {
                        const response = await fetch('/api/providers', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(providerData)
                        });

                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(errorData.error || 'Failed to add provider');
                        }

                        providerForm.reset();
                        fetchProviders(); // Refresh the list after adding
                        showMessage('Provider added successfully!', 'success');
                    } catch (error) {
                        console.error('Error adding provider:', error);
                        showMessage(`Error: ${error.message}`, 'error');
                    }
                });

                // Initial fetch of providers
                fetchProviders();
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)


@app.route('/api/providers', methods=['GET'])
def list_providers():
    """API endpoint to get all providers."""
    providers = get_providers_from_db()
    return jsonify(providers)


@app.route('/api/providers', methods=['POST'])
def create_provider():
    """API endpoint to add a new provider."""
    data = request.get_json()
    if not data or not all(k in data for k in ["name", "serviceType", "description", "contactInfo"]):
        return jsonify({"error": "Missing provider data"}), 400

    new_doc_id = add_provider_to_db(data)
    if new_doc_id:
        return jsonify({"id": new_doc_id, "message": "Provider added successfully"}), 201
    else:
        return jsonify({"error": "Failed to add provider"}), 500


if __name__ == "__main__":
    # The 'host="0.0.0.0"' makes the server accessible from outside the local machine,
    # which is necessary for this collaborative environment.
    app.run(host="0.0.0.0", port=5000, debug=True)

