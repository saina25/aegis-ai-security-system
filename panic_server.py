"""
This is the new "Panic Button Server" (v2).
It now uses `ngrok` to create a public URL that bypasses firewalls.
"""

from flask import Flask, send_from_directory, jsonify
from pathlib import Path
import os
import logging
import socket
from pyngrok import ngrok  # <-- NEW: Import ngrok

# Define the signal file that main.py will look for
SIGNAL_FILE = "panic_trigger.signal"

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)

app = Flask(__name__)

# --- NEW: Setup the ngrok tunnel ---
# This line creates a public URL that "tunnels" to our local port 5000
public_url = ngrok.connect(5000)
print("\n" + "="*50)
print("AEGIS Panic Button Server (v2 with ngrok)")
print(f"Signal file: '{SIGNAL_FILE}'")
print("\n--- HOW TO CONNECT ---")
print("1. Make sure your phone has internet.")
print("2. Open your phone's browser and go to THIS PUBLIC URL:")
print(f"   {public_url}")
print("="*50 + "\n")
# ------------------------------------

@app.route('/')
def index():
    """Serves the main panic_page.html file."""
    log.info(f"Phone connected from {public_url}. Serving panic_page.html")
    return send_from_directory('.', 'panic_page.html')

@app.route('/trigger', methods=['POST'])
def trigger_alert():
    """
    This is the endpoint the panic button (phone) calls.
    It creates the signal file that the main Streamlit app is looking for.
    """
    try:
        Path(SIGNAL_FILE).touch()
        log.info(f"*** PANIC BUTTON PRESSED ***")
        log.info(f"Signal file '{SIGNAL_FILE}' created.")
        return jsonify({"status": "success", "message": "Alert triggered!"}), 200
    except Exception as e:
        log.error(f"Error creating signal file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Clean up any old signal file on startup
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)
        print(f"Removed old signal file: {SIGNAL_FILE}")
    
    # Host on 0.0.0.0 to make it accessible
    app.run(host='0.0.0.0', port=5000)

