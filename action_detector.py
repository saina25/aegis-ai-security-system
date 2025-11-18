import cv2
from ultralytics import YOLO
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

class ActionDetector:
    """
    V7: Replaced all rule-based logic with a true,
    AI-based, multi-person action recognition system.
    
    This uses two models:
    1. YOLOv8n (Object Detection) to find all people.
    2. Siglip (Image Classification) to classify the action of each person.
    """
    def __init__(self):
        print("[ActionDetector] Loading YOLOv8 person detector...")
        self.person_detector = YOLO("yolov8n.pt") # Standard person detector
        
        print("[ActionDetector] Loading Hugging Face action classifier...")
        # This will auto-download the model from Hugging Face
        model_name = "prithivMLmods/Human-Action-Recognition"
        self.action_processor = AutoImageProcessor.from_pretrained(model_name)
        self.action_model = SiglipForImageClassification.from_pretrained(model_name)
        
        # Get the class names from the model's config
        self.action_labels = self.action_model.config.id2label
        print("[ActionDetector] All models loaded successfully.")

    def process_frame(self, frame):
        """
        Runs full AI-based action recognition on a single frame.
        
        Returns:
            person_boxes (list): List of [x1, y1, x2, y2] for all people.
            action_labels (list): List of string labels (e.g., "fighting") for each person.
            action_confs (list): List of confidence scores (0.0-1.0) for each action.
        """
        
        # --- Branch 1: Find all people ---
        # We only care about class 0 (person)
        results = self.person_detector.predict(frame, classes=[0], verbose=False, conf=0.5)
        
        person_boxes = []
        action_labels = []
        action_confs = []
        
        # Get the first (and only) result object
        result = results[0]
        
        if len(result.boxes) == 0:
            # No people found, return empty lists
            return [], [], []

        # Get all person bounding boxes in [x1, y1, x2, y2] format
        person_boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)

        # --- Branch 2: Classify action for each person ---
        for box in person_boxes_xyxy:
            x1, y1, x2, y2 = box
            
            # Crop the person from the frame
            # Add padding to give the model context
            padding = 20
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame.shape[1], x2 + padding)
            crop_y2 = min(frame.shape[0], y2 + padding)
            
            person_crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Convert from OpenCV (BGR) to PIL (RGB)
            pil_image = Image.fromarray(cv2.cvtColor(person_crop_img, cv2.COLOR_BGR2RGB))
            
            # Process the image and run the classifier
            inputs = self.action_processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.action_model(**inputs)
            
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
            
            # Get the top prediction
            top_conf, top_idx = torch.max(probs, 0)
            
            # Store the results
            person_boxes.append([x1, y1, x2, y2])
            action_labels.append(self.action_labels[top_idx.item()])
            action_confs.append(top_conf.item())

        return person_boxes, action_labels, action_confs