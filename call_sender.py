import streamlit as st
from twilio.rest import Client
import os

def make_emergency_call(alert_time, trigger_reason="An automated alert"):
    """
    Uses Twilio to make an automated, real-time voice call.
    Now includes a 'trigger_reason' for more context.
    """
    try:
        # --- Get All Secrets ---
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        twilio_phone_number = st.secrets["TWILIO_PHONE_NUMBER"]
        recipient_phone_number = st.secrets["DEMO_RECIPIENT_NUMBER"]
        
        location_name = st.secrets.get("LOCATION", "An unspecified location")

        # --- Initialize Twilio Client ---
        client = Client(account_sid, auth_token)

        # --- Define the new, more descriptive message ---
        call_message = f"""
        <Response>
            <Say voice="alice" language="en-IN">
                This is an automated emergency alert from the AEGIS security system.
                {trigger_reason} has been detected at {location_name} at {alert_time}.
                Repeating. This is a confirmed emergency at {location_name}. Please respond immediately.
            </Say>
        </Response>
        """

        # --- Make the Call ---
        call = client.calls.create(
            twiml=call_message,
            to=recipient_phone_number,
            from_=twilio_phone_number
        )
        
        return True

    except Exception as e:
        # Show a detailed error in the app if the call fails
        st.error(f"Failed to make emergency call: {e}")
        return False
