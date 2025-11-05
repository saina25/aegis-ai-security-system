import cv2
import numpy as np
from ultralytics import YOLO

# --- CLASS NAMES TO DETECT ---
TARGET_CLASSES = ["gun", "knife"] 

# --- MODEL CONFIG ---
MODEL_PATH = "yolov8n-weapons.pt"  # <-- UPDATE THIS

# --- DETECTION PARAMETERS ---
# --- CHANGED: Raised threshold to fix "hand-as-knife" ---
CONF_THRESHOLD = 0.4  # Was 0.5. Raise this if you get false positives.

class WeaponDetector:
    """
    Encapsulates the YOLOv8 object detection model
    to find 'gun', 'knife', or other defined weapons.
    """
    def __init__(self):
        try:
            print("[WeaponDetector] Loading YOLOv8 model...")
            self.model = YOLO(MODEL_PATH)
            dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
            self.model.predict(dummy_img, conf=0.5, verbose=False)
            print("[WeaponDetector] Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            print(f"Please make sure the file '{MODEL_PATH}' exists")
            raise e

    def process_frame(self, frame):
        """
        Runs weapon detection on a single frame.
        
        Returns:
            is_weapon_present (bool): True if any target class is found.
            weapon_boxes (list): List of [x, y, w, h] boxes for drawing.
            class_names (list): List of detected class names.
            confidences (list): List of confidence scores (0.0 to 1.0).
        """
        
        results = self.model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        is_weapon_present = False
        final_weapon_boxes = []
        final_class_names = []
        final_confidences = [] # --- NEW: List to store confidences ---

        for box in result.boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            if class_name in TARGET_CLASSES:
                is_weapon_present = True
                
                # Get box coordinates
                bbox_xywh = box.xywh.cpu().numpy().astype(int)[0]
                x = int(bbox_xywh[0] - bbox_xywh[2] / 2)
                y = int(bbox_xywh[1] - bbox_xywh[3] / 2)
                w = int(bbox_xywh[2])
                h = int(bbox_xywh[3])
                
                # --- NEW: Get confidence score ---
                conf = float(box.conf.cpu().numpy()[0])
                
                final_weapon_boxes.append([x, y, w, h])
                final_class_names.append(class_name.upper())
                final_confidences.append(conf) # --- NEW: Add to list ---

        # --- CHANGED: Return confidences as well ---
        return is_weapon_present, final_weapon_boxes, final_class_names, final_confidences