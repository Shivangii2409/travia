from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ===========================
# Configure Gemini API
# ===========================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.0-pro')

# ===========================
# Load ML model & encoders
# ===========================
features = [
    'Name_x', 'Type', 'Preferences', 'Gender',
    'NumberOfAdults', 'NumberOfChildren', 'NumberOfDays'
]

model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# ===========================
# Load CSVs
# ===========================
destinations_df = pd.read_csv("Expanded_Destinations.csv")
userhistory_df = pd.read_csv("Final_Updated_Expanded_UserHistory.csv")

# Collaborative filtering setup
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')
user_item_matrix.fillna(0, inplace=True)
user_similarity = cosine_similarity(user_item_matrix)

# Simple in-memory user store
users = {}

# Store chat history per user
chat_histories = {}


# ===========================
# Helper functions
# ===========================
def collaborative_recommend(user_similarity, user_item_matrix, destinations_df):
    similar_users_idx = np.argsort(user_similarity.mean(axis=0))[::-1][1:6]
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
    recommended_ids = similar_user_ratings.sort_values(ascending=False).head(5).index
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_ids)][
        ['DestinationID', 'Name', 'Type', 'Popularity', 'BestTimeToVisit']
    ]
    return recommendations.to_dict('records') if not recommendations.empty else []


def safe_encode(feature, value):
    if feature in label_encoders:
        le = label_encoders[feature]
        if value not in le.classes_:
            le.classes_ = np.append(le.classes_, value)
        return le.transform([value])[0]
    return value


def recommend_destinations(user_input, model, label_encoders, features, destinations_df):
    encoded_input = {}
    for feature in features:
        if feature in user_input:
            encoded_input[feature] = safe_encode(feature, user_input[feature])
    input_df = pd.DataFrame([encoded_input])
    predicted_popularity = model.predict(input_df)[0]
    return predicted_popularity


def get_travel_context():
    """Provide context about available destinations to Gemini"""
    dest_list = destinations_df[['Name', 'Type', 'BestTimeToVisit']].head(20).to_string(index=False)
    return f"""You are a helpful travel assistant for a Tourism Recommendation System. 
    You help users plan trips, answer questions about destinations, and provide travel advice.

    Here are some popular destinations in our database:
    {dest_list}

    Be friendly, informative, and helpful. Keep responses concise but informative."""


# ===========================
# ROUTES
# ===========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash("Please fill all fields!", "danger")
            return redirect(url_for('register'))

        if username in users:
            flash("Username already exists!", "warning")
        else:
            users[username] = password
            flash("Registered successfully! Please login.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username] == password:
            session['user'] = username
            # Initialize chat history for user
            if username not in chat_histories:
                chat_histories[username] = []
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials!", "danger")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if 'user' not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for('login'))

    if request.method == "POST":
        user_input = {
            'Name_x': session['user'],
            'Type': request.form['type'],
            'Preferences': request.form['preferences'],
            'Gender': request.form['gender'],
            'NumberOfAdults': request.form['adults'],
            'NumberOfChildren': request.form['children'],
            'NumberOfDays': request.form['days']
        }

        recommended_destinations = collaborative_recommend(user_similarity, user_item_matrix, destinations_df)
        predicted_popularity = recommend_destinations(user_input, model, label_encoders, features, destinations_df)

        return render_template(
            'recommendation.html',
            recommended_destinations=recommended_destinations,
            predicted_popularity=predicted_popularity,
            user=session['user']
        )

    return render_template('recommendation.html', user=session['user'])


# ===========================
# CHATBOT ROUTES
# ===========================
# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle chatbot conversations using Gemini API"""
#     if 'user' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
#
#     try:
#         user_message = request.json.get('message', '').strip()
#         username = session['user']
#
#         if not user_message:
#             return jsonify({'error': 'Empty message'}), 400
#
#         # Get or initialize chat history
#         if username not in chat_histories:
#             chat_histories[username] = []
#
#         # Create chat session with context
#         chat = gemini_model.start_chat(history=[])
#
#         # Add travel context to first message
#         context_prompt = f"{get_travel_context()}\n\nUser: {user_message}"
#
#         # Generate response
#         response = chat.send_message(context_prompt)
#         bot_response = response.text
#
#         # Store in history
#         chat_histories[username].append({
#             'user': user_message,
#             'bot': bot_response
#         })
#
#         # Keep only last 10 exchanges
#         if len(chat_histories[username]) > 10:
#             chat_histories[username] = chat_histories[username][-10:]
#
#         return jsonify({
#             'response': bot_response,
#             'success': True
#         })
#
#     except Exception as e:
#         print(f"Chatbot error: {e}")
#         return jsonify({
#             'error': 'Sorry, I encountered an error. Please try again.',
#             'success': False
#         }), 500
#
#
# @app.route('/chat-history')
# def get_chat_history():
#     """Get chat history for current user"""
#     if 'user' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
#
#     username = session['user']
#     history = chat_histories.get(username, [])
#     return jsonify({'history': history})
#
#
# @app.route('/clear-chat', methods=['POST'])
# def clear_chat():
#     """Clear chat history for current user"""
#     if 'user' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
#
#     username = session['user']
#     chat_histories[username] = []
#     return jsonify({'success': True})

# ===========================
# CHATBOT ROUTES (REVISED)
# ===========================

def format_history_for_gemini(history):
    """Converts our chat history format to Gemini's format."""
    gemini_history = []
    for message in history:
        gemini_history.append({'role': 'user', 'parts': [message['user']]})
        gemini_history.append({'role': 'model', 'parts': [message['bot']]})
    return gemini_history


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot conversations using Gemini API"""
    print("\n--- NEW CHAT REQUEST ---")  # DEBUG

    if 'user' not in session:
        print("DEBUG: User not in session. Aborting.")  # DEBUG
        return jsonify({'error': 'Not logged in'}), 401

    try:
        user_message = request.json.get('message', '').strip()
        username = session['user']
        print(f"DEBUG: Received message: '{user_message}' from user: '{username}'")  # DEBUG

        if not user_message:
            print("DEBUG: Empty message received. Aborting.")  # DEBUG
            return jsonify({'error': 'Empty message'}), 400

        user_history = chat_histories.get(username, [])
        gemini_formatted_history = format_history_for_gemini(user_history)

        print("DEBUG: Starting chat session with Gemini...")  # DEBUG
        chat_session = gemini_model.start_chat(history=gemini_formatted_history)

        prompt_to_send = user_message
        if not user_history:
            prompt_to_send = f"{get_travel_context()}\n\nUser: {user_message}"

        print("DEBUG: Sending message to Gemini API...")  # DEBUG
        response = chat_session.send_message(prompt_to_send)
        print("DEBUG: Received response from Gemini API.")  # DEBUG

        if not response.parts:
            print(f"DEBUG: Gemini response was blocked. Reason: {response.prompt_feedback}")  # DEBUG
            bot_response = "I'm sorry, I can't respond to that. Let's talk about something else."
        else:
            bot_response = response.text
            print(f"DEBUG: Gemini response text: '{bot_response[:80]}...'")  # DEBUG

        user_history.append({
            'user': user_message,
            'bot': bot_response
        })

        chat_histories[username] = user_history[-10:]

        print("DEBUG: Sending successful JSON response to frontend.")  # DEBUG
        return jsonify({
            'response': bot_response,
            'success': True
        })

    except Exception as e:
        # This part is VERY IMPORTANT. If there's an error, it will be printed here.
        print(f"!!!!!!!!!!!!!! FATAL CHATBOT ERROR !!!!!!!!!!!!!!")
        print(f"Chatbot error: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return jsonify({
            'error': f'Sorry, an internal error occurred. Please check server logs.',
            'success': False
        }), 500

# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle chatbot conversations using Gemini API"""
#     if 'user' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
#
#     try:
#         user_message = request.json.get('message', '').strip()
#         username = session['user']
#
#         if not user_message:
#             return jsonify({'error': 'Empty message'}), 400
#
#         user_history = chat_histories.get(username, [])
#         gemini_formatted_history = format_history_for_gemini(user_history)
#         chat_session = gemini_model.start_chat(history=gemini_formatted_history)
#
#         prompt_to_send = user_message
#         if not user_history:
#             prompt_to_send = f"{get_travel_context()}\n\nUser: {user_message}"
#
#         # Generate response
#         response = chat_session.send_message(prompt_to_send)
#
#         # === NEW: Check for safety blocks ===
#         if not response.parts:
#             # This can happen if the response was blocked
#             print(f"Chatbot safety block: {response.prompt_feedback}")
#             bot_response = "I'm sorry, I can't respond to that. Let's talk about something else."
#         else:
#             bot_response = response.text
#         # === End of new block ===
#
#         user_history.append({
#             'user': user_message,
#             'bot': bot_response
#         })
#
#         chat_histories[username] = user_history[-10:]
#
#         return jsonify({
#             'response': bot_response,
#             'success': True
#         })
#
#     except Exception as e:
#         print(f"Chatbot error: {e}")
#         return jsonify({
#             'error': f'Sorry, I encountered an error. Please check server logs for details.',
#             'success': False
#         }), 500
#
# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle chatbot conversations using Gemini API"""
#     if 'user' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
#
#     try:
#         user_message = request.json.get('message', '').strip()
#         username = session['user']
#
#         if not user_message:
#             return jsonify({'error': 'Empty message'}), 400
#
#         # Get or initialize chat history
#         user_history = chat_histories.get(username, [])
#
#         # Format for Gemini
#         gemini_formatted_history = format_history_for_gemini(user_history)
#
#         # Start chat session with the user's existing history
#         chat_session = gemini_model.start_chat(history=gemini_formatted_history)
#
#         # Prepend context only if it's the first message of the session
#         prompt_to_send = user_message
#         if not user_history:
#             print("This is the first message. Adding context.")
#             prompt_to_send = f"{get_travel_context()}\n\nUser: {user_message}"
#
#         # Generate response
#         response = chat_session.send_message(prompt_to_send)
#         bot_response = response.text
#
#         # Store the new exchange in our history
#         user_history.append({
#             'user': user_message,
#             'bot': bot_response
#         })
#
#         # Keep only last 10 exchanges to manage memory and token count
#         if len(user_history) > 10:
#             chat_histories[username] = user_history[-10:]
#         else:
#             chat_histories[username] = user_history
#
#         return jsonify({
#             'response': bot_response,
#             'success': True
#         })
#
#     except Exception as e:
#         print(f"Chatbot error: {e}")
#         # Be more specific for debugging if you can
#         return jsonify({
#             'error': f'Sorry, I encountered an error: {str(e)}',
#             'success': False
#         }), 500


# The other chat routes (/chat-history, /clear-chat) are fine as they are.
# Just ensure they are placed after the new /chat route.

@app.route('/chat-history')
def get_chat_history():
    """Get chat history for current user"""
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    username = session['user']
    history = chat_histories.get(username, [])
    return jsonify({'history': history})


@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    """Clear chat history for current user"""
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    username = session['user']
    if username in chat_histories:
        chat_histories[username] = []
    return jsonify({'success': True})

if __name__ == '__main__':
  app.run(debug=True)
