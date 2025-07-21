import socketio
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime
from chatbot.core import process_message

# Initialize Flask app
app = Flask(__name__)

# Initialize Socket.IO server
# IMPORTANT: Adjust cors_allowed_origins to your React Native app's IP/port
# For development, you can use "*" but for production, specify exact origins.
sio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.route('/')
def index():
    return jsonify({"status": "Socket.IO server is running"})

@sio.event
def connect():
    """Event handler for new client connections."""
    print(f"Client connected: {request.sid}")
    # You can send a welcome message immediately upon connection
    emit('chat_message', {'sender': 'Bot', 'message': 'Hello! How can I help you today?'}, room=request.sid)

@sio.event
def disconnect(sid):
    """Event handler for client disconnections."""
    print(f"Client disconnected: {sid}")

@sio.event
def user_message(sid, data):
    """
    Event handler for messages received from the client.
    'data' is expected to be an object like: { 'message': 'User\'s input' }
    """
    user_input = data.get('message', '').strip()
    print(f"Message from {sid}: {user_input}")

    # bot_response = process_message(user_message=user_input)

    # Simulate chatbot processing
    bot_response = "I'm sorry, I don't understand that yet."
    if "hello" in user_input.lower():
        bot_response = "Hi there! I'm your GrindHub AI assistant."
    elif "schedule" in user_input.lower():
        bot_response = "I can help you with your schedule. What date are you looking for?"
    elif "assignment" in user_input.lower():
        bot_response = "For assignments, you can check the 'Your Assignments' section on your homepage. What specific assignment are you curious about?"
    elif "thank you" in user_input.lower():
        bot_response = "You're welcome! Happy to help."

    # Send the bot's response back to the client
    emit('chat_message', {'sender': 'Bot', 'message': bot_response}, room=sid)

if __name__ == '__main__':
    # This block is primarily for local direct execution,
    # but Gunicorn will handle running `app` directly.
    # If you remove this, ensure you test with Gunicorn locally first.
    # For Railway, Gunicorn will find and run 'app'.
    print("Running locally with Flask-SocketIO's development server. Use Gunicorn for production.")
    sio.run(app, host='0.0.0.0', port=5000, debug=True)