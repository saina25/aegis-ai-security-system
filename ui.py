import streamlit as st
from call_sender import make_emergency_call
import time 

def setup_layout():
    """Sets up the main layout of the Streamlit page."""
    st.set_page_config(layout="wide")
    st.title("AEGIS - 3-Branch Interaction Engine")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        video_placeholder = st.empty()
    with col2:
        st.subheader("System Status")
        status_placeholder = st.empty()
        st.subheader("Alerts")
        alert_placeholder = st.empty()
        
    return video_placeholder, status_placeholder, alert_placeholder

def update_status(placeholder, alert_level):
    """Updates the system status panel based on the P-level."""
    if alert_level == "P1":
        placeholder.error("P1 CRITICAL (Weapon Interaction)", icon="🚨")
    elif alert_level == "P2":
        placeholder.warning("P2 HIGH (Potential Threat)", icon="⚠️")
    else:
        placeholder.success("P0 (Monitoring... All clear)", icon="✅")

def render_p1_critical_alert(placeholder):
    """
    Displays the P1 Critical Alert.
    This now INCLUDES the automated call logic.
    """
    with placeholder.container():
        st.error(f"P1 CRITICAL ALERT at {st.session_state.alert_time}", icon="🚨")
        st.write(f"Reason: {st.session_state.trigger_reason}")
        
        # --- Automated Call ---
        with st.spinner("Automated 'P1' alert... Placing call to authorities..."):
            make_emergency_call(
                alert_time=st.session_state.alert_time,
                trigger_reason=st.session_state.trigger_reason
            )
        st.success("Automated call has been dispatched.", icon="✔️")
        
        st.image(st.session_state.alert_image, caption="Alert Snapshot")
        
        # --- FIX: Reset Button Logic ---
        if st.button("Reset System", key="p1_reset"):
            st.session_state.alert_triggered_p1 = False
            st.session_state.alert_image = None
            st.session_state.trigger_reason = ""
            st.session_state.person_tracker = {}
            st.rerun()

def render_p2_verification_dashboard(placeholder):
    """
    Displays the P2 Verification dashboard for human-in-the-loop.
    """
    with placeholder.container():
        st.warning(f"P2 HIGH ALERT at {st.session_state.alert_time}", icon="⚠️")
        st.write(f"Reason: {st.session_state.trigger_reason}")
        st.image(st.session_state.alert_image, caption="Alert Snapshot")
        
        st.info("Please verify the situation. Is this a real emergency?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Emergency", use_container_width=True, type="primary", key="p2_confirm"):
                # 1. Escalate to P1
                st.session_state.alert_triggered_p2 = False
                st.session_state.alert_triggered_p1 = True
                # Update reason for the P1 call
                st.session_state.trigger_reason = f"Manually Confirmed P2 Alert ({st.session_state.trigger_reason})"
                st.session_state.person_tracker = {}
                st.rerun()

        with col2:
            # --- FIX: Dismiss Button Logic ---
            if st.button("❌ Dismiss as False Alarm", use_container_width=True, key="p2_dismiss"):
                st.session_state.alert_triggered_p2 = False
                st.session_state.alert_image = None
                st.session_state.trigger_reason = ""
                st.session_state.person_tracker = {}
                st.rerun()

# --- REMOVED ---
# def render_demo_controls(placeholder):
#     ...

