import cv2
import streamlit as st
from datetime import datetime
import os

# --- NEW: Import the new AI "Brains" ---
from yolov8_weapon_detector import WeaponDetector
from action_detector import ActionDetector # This is our new V7 AI Classifier version

# --- OLD: Keep all the UI and Action components ---
import ui
from call_sender import make_emergency_call 

# --- REMOVED: All MediaPipe/YOLO-Pose drawing helpers are no longer needed ---

# --- CONSTANTS (from old system) ---
SIGNAL_FILE = "panic_trigger.signal"
DETECTION_THRESHOLD = 2 # 3 consecutive frames

# ----------------------------------------------------------------------
# --- NEW: Simplified Drawing helpers ---
# ----------------------------------------------------------------------

def draw_boxes(image, boxes, class_names, confs, color):
    """Helper function to draw weapon boxes with confidence scores."""
    for (x, y, w, h), class_name, conf in zip(boxes, class_names, confs):
        x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        cv2.rectangle(image, (x, y), (x2, y2), color, 3)
        label = f"{class_name} ({conf*100:.0f}%)"
        cv2.putText(image, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

def draw_person_boxes(image, boxes, labels, confs, p_color):
    """
    Draws person boxes with their AI-CLASSIFIED action label.
    """
    for box, label, conf in zip(boxes, labels, confs):
        x1, y1, x2, y2 = box
        
        # Create the new label, e.g., "PERSON (fighting 88%)"
        ai_label = f"PERSON ({label.upper()} {conf*100:.0f}%)"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), p_color, 2)
        cv2.putText(image, ai_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, p_color, 2)


def check_overlap(boxA, boxB):
    """Checks if two bounding boxes [x1, y1, x2, y2] overlap."""
    if boxA[2] < boxB[0] or boxB[2] < boxA[0]:
        return False
    if boxA[3] < boxB[1] or boxB[3] < boxA[1]:
        return False
    return True

# ----------------------------------------------------------------------
# --- Streamlit Cached Functions (Now loading the *new* models) ---
# ----------------------------------------------------------------------

@st.cache_resource
def get_camera():
    """Use st.cache_resource to get the camera object."""
    print("Attempting to initialize camera...")
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera initialized successfully on index {index}.")
            return cap
    st.error("Failed to open any camera (tried indices 0, 1, 2). Is it in use or disconnected? Try refreshing.")
    return None

@st.cache_resource
def get_weapon_detector():
    """Loads the YOLOv8 Weapon Detector model only once."""
    print("Loading YOLOv8 Weapon Detector...")
    return WeaponDetector()

@st.cache_resource
def get_action_detector():
    """Loads the new AI Action Classifier model."""
    print("Loading AI Action Classifier...")
    return ActionDetector()


# ----------------------------------------------------------------------
# --- MAIN APPLICATION ---
# ----------------------------------------------------------------------

def main():
    video_placeholder, status_placeholder, alert_placeholder = ui.setup_layout()

    try:
        yolo_model = get_weapon_detector()
        pose_model = get_action_detector() # This now loads the new AI detector
    except Exception as e:
        st.error(f"FATAL: Failed to initialize AI models: {e}")
        st.error("Have you run 'pip install transformers Pillow'?")
        st.error("The models will download automatically on first run.")
        st.stop()
            
    cap = get_camera()
    if cap is None:
        st.stop()

    # --- Initialize session state variables (UNCHANGED) ---
    if "alert_triggered_p1" not in st.session_state:
        st.session_state.alert_triggered_p1 = False
    if "alert_triggered_p2" not in st.session_state:
        st.session_state.alert_triggered_p2 = False
    if "alert_image" not in st.session_state:
        st.session_state.alert_image = None
    if "alert_time" not in st.session_state:
        st.session_state.alert_time = ""
    if "trigger_reason" not in st.session_state:
        st.session_state.trigger_reason = "An unknown alert"
    if "p1_counter" not in st.session_state:
        st.session_state.p1_counter = 0
    if "p2_counter" not in st.session_state:
        st.session_state.p2_counter = 0

    # --- Panic Button Check (UNCHANGED) ---
    if os.path.exists(SIGNAL_FILE):
        print("PANIC SIGNAL DETECTED!")
        os.remove(SIGNAL_FILE) 
        ret, frame = cap.read()
        if ret:
            st.session_state.alert_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.trigger_reason = "a manual 'Panic Button' signal"
        st.session_state.alert_triggered_p1 = True
        st.rerun()

    # --- State Management (UNCHANGED) ---
    if st.session_state.alert_triggered_p1:
        ui.render_p1_critical_alert(alert_placeholder)
        if st.session_state.alert_image is not None:
            video_placeholder.image(st.session_state.alert_image, channels="RGB", use_container_width=True)
        return 
    if st.session_state.alert_triggered_p2:
        ui.render_p2_verification_dashboard(alert_placeholder)
        if st.session_state.alert_image is not None:
            video_placeholder.image(st.session_state.alert_image, channels="RGB", use_container_width=True)
        return 

    # --- State 3: Normal Monitoring (Main Loop) ---
    ui.update_status(status_placeholder, "P0") # Default status

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam feed failed.")
            break 

        frame = cv2.flip(frame, 1)
        frame_for_snapshot = frame.copy() 
        
        # --- Run AI Models ---
        is_weapon, weapon_boxes, weapon_names, weapon_confs = yolo_model.process_frame(frame)
        
        # --- NEW: Get AI-based person and action results ---
        person_boxes_xyxy, action_labels, action_confs = pose_model.process_frame(frame)

        # --- Branch 3: Interaction Engine ---
        is_interaction = False
        active_weapon_name = ""
        
        if is_weapon and len(person_boxes_xyxy) > 0:
            for w_box, w_name in zip(weapon_boxes, weapon_names):
                w_box_x1y1x2y2 = [w_box[0], w_box[1], w_box[0] + w_box[2], w_box[1] + w_box[3]]
                for p_box in person_boxes_xyxy: # p_box is already [x1, y1, x2, y2]
                    if check_overlap(w_box_x1y1x2y2, p_box):
                        is_interaction = True
                        active_weapon_name = w_name.upper()
                        break
                if is_interaction:
                    break
        
        # --- P-Level "Smart Filter" (Now uses AI labels) ---
        alert_level = "P0"
        p1_reason = ""
        p2_reason = ""
        
                # --- NEW: Check for AI labels *with high confidence* ---
        is_fighting_action = False
        FIGHTING_CONF_THRESHOLD = 0.6 # Only trigger if 60% sure

        for label, conf in zip(action_labels, action_confs):
            if label == "fighting" and conf > FIGHTING_CONF_THRESHOLD:
                is_fighting_action = True
                break # Found one, no need to keep looking
        
        if is_interaction and "KNIFE" in active_weapon_name and is_fighting_action:
            alert_level = "P1"
            p1_reason = "Automated 'P1 Critical' (AI: FIGHTING with KNIFE)"
        elif is_interaction and "GUN" in active_weapon_name and is_fighting_action:
            alert_level = "P1"
            p1_reason = "Automated 'P1 Critical' (AI: FIGHTING with GUN)"
        elif alert_level == "P0" and is_interaction and ("GUN" in active_weapon_name or "KNIFE" in active_weapon_name):
            alert_level = "P2"
            p2_reason = f"Automated 'P2 High' (AI: Person CARRYING {active_weapon_name})"
        elif alert_level == "P0" and is_fighting_action:
            alert_level = "P2"
            p2_reason = "AI Detected: FIGHTING" # The reason is now direct from the AI
            
        # --- Drawing Logic (V5 - AI Classifier) ---
        P0_COLOR = (0, 255, 0)
        P2_COLOR = (0, 165, 255)
        P1_COLOR = (0, 0, 255)
        
        person_color = P0_COLOR
        if alert_level == "P1":
            person_color = P1_COLOR
        elif alert_level == "P2":
            person_color = P2_COLOR

        # 1. Draw ALL Person Boxes with their AI Action Label
        draw_person_boxes(frame, person_boxes_xyxy, action_labels, action_confs, person_color)

        # 2. Draw Weapon Boxes
        weapon_color = (255, 255, 0) # Yellow (Ignored)
        if alert_level == "P1":
            weapon_color = P1_COLOR
        elif alert_level == "P2" and "CARRYING" in p2_reason:
            weapon_color = P2_COLOR
        if is_weapon:
            draw_boxes(frame, weapon_boxes, weapon_names, weapon_confs, weapon_color)
        
        # --- Update UI ---
        ui.update_status(status_placeholder, alert_level)
        rgb_annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_annotated_frame, use_container_width=True)

        
        # --- Consistency counter logic (UNCHANGED) ---
        if alert_level == "P1":
            st.session_state.p1_counter += 1
            st.session_state.p2_counter = 0
        elif alert_level == "P2":
            st.session_state.p1_counter = 0
            st.session_state.p2_counter += 1
        else: # P0
            st.session_state.p1_counter = 0
            st.session_state.p2_counter = 0
        
        if st.session_state.p1_counter >= DETECTION_THRESHOLD:
            print(f"Consistent P1 detected! Reason: {p1_reason}")
            st.session_state.alert_image = cv2.cvtColor(frame_for_snapshot, cv2.COLOR_BGR2RGB)
            st.session_state.alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.trigger_reason = p1_reason 
            st.session_state.alert_triggered_p1 = True
            st.session_state.p1_counter = 0 
            st.rerun()
        elif st.session_state.p2_counter >= DETECTION_THRESHOLD:
            print(f"Consistent P2 detected! Reason: {p2_reason}")
            st.session_state.alert_image = cv2.cvtColor(frame_for_snapshot, cv2.COLOR_BGR2RGB)
            st.session_state.alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.trigger_reason = p2_reason 
            st.session_state.alert_triggered_p2 = True
            st.session_state.p2_counter = 0 
            st.rerun()
        

if __name__ == "__main__":
    try:
        main()
    finally:
        if os.path.exists(SIGNAL_FILE):
            os.remove(SIGNAL_FILE)
        print("\nAEGIS system shutting down. Signal file cleaned up.")