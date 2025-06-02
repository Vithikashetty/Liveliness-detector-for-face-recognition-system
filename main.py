from time import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

####################################
# Configuration
outputFolderPath = 'Dataset/DataCollect'
confidence_threshold = 0.4
save = True
blurThreshold = 30  # Lowered from 50 to 30
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
classes = ["fake", "real"]
####################################

def download_models():
    """Download required models if they don't exist"""
    import requests
    from pathlib import Path
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download face detection model if not present
    face_model_path = "models/yolov8n-face.pt"
    if not os.path.exists(face_model_path):
        print("Downloading YOLOv8 face detection model...")
        url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
        r = requests.get(url, allow_redirects=True)
        with open(face_model_path, 'wb') as f:
            f.write(r.content)
        print("Face detection model downloaded!")

def calculate_blur(image):
    """Calculate the blurriness of an image using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_eye_blink(face_img):
    """Check for eye blinking (a sign of liveliness)"""
    # This is a simplified eye detection - in a real implementation, you'd use
    # a more sophisticated eye detection and blink detection system
    # We'll use Haar cascade for basic eye detection
    try:
        # Create eye detector
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If multiple eyes detected, more likely a real face
        return len(eyes) >= 1
    except Exception as e:
        print(f"Error in eye detection: {e}")
        return False

def check_specular_highlights(face_img):
    """Check for natural specular highlights on skin (common in real faces)"""
    # Convert to HSV
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Apply threshold to find highlight regions
    _, binary = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours of highlight regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Real faces typically have multiple small highlights rather than large reflections
    highlight_count = len(contours)
    
    # Check if there are a reasonable number of small highlights
    if highlight_count > 1 and highlight_count < 15:
        # Check size of highlights
        areas = [cv2.contourArea(c) for c in contours]
        largest_area = max(areas) if areas else 0
        total_area = face_img.shape[0] * face_img.shape[1]
        
        # Natural highlights are usually small
        if largest_area < (total_area * 0.05):
            return True
    
    return False

def check_color_variance(face_img):
    """Check for natural skin color variance"""
    # Convert to YCrCb (better for skin detection)
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Real faces have natural variance in chrominance channels
    cr_std = np.std(cr)
    cb_std = np.std(cb)
    
    # Realistic skin tones have moderate variance
    # Too high = artificial coloring, too low = flat photo/screen
    return 4 < cr_std < 25 and 4 < cb_std < 20

def is_real_face(face_img, blur_value):
    """Determine if a face is real or from a screen/photo based on multiple metrics"""
    # Check for basic image validity
    if face_img is None or face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
        print("Image too small or invalid")
        return False
    
    # Store individual test results for logging
    results = {}
    debug_info = {}
    
    # Check 1: Blur test - real faces shouldn't be too blurry
    results["blur_test"] = (blur_value > blurThreshold)
    debug_info["blur_value"] = blur_value
    
    if not results["blur_test"]:
        print(f"Failed blur test: {blur_value} < {blurThreshold}")
        return False
    
    # Check 2: Eye detection - real faces should have detectable eyes
    results["eye_test"] = check_eye_blink(face_img)
    
    # Check 3: Natural skin color variance
    results["color_variance_test"] = check_color_variance(face_img)
    
    # Check 4: Natural specular highlights
    results["specular_test"] = check_specular_highlights(face_img)
    
    # Scoring system - needs to pass at least 2 of the 3 advanced tests
    score = sum([
        results["eye_test"], 
        results["color_variance_test"], 
        results["specular_test"]
    ])
    
    # Print detailed debug info
    print(f"Test results - Blur: Pass, Eyes: {results['eye_test']}, " +
          f"Color: {results['color_variance_test']}, Specular: {results['specular_test']}")
    
    # For a face to be considered real, it must pass the blur test AND
    # at least 2 of the 3 advanced tests
    is_real = score >= 2
    
    if is_real:
        print("All tests passed - detected as real face")
    else:
        print(f"Failed advanced tests - score: {score}/3")
    
    return is_real

def main():
    # Download the face detection model if needed
    download_models()
    
    # Load YOLOv8 face detection model
    try:
        face_model = YOLO("models/yolov8n-face.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have installed ultralytics package: pip install ultralytics")
        return
    
    # Set up webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, camWidth)
    cap.set(4, camHeight)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    prev_frame_time = 0
    manual_class_id = None  # None means auto-detection, otherwise use manual override
    
    print("Press 'f' for Fake, 'r' for Real, 'a' for Auto detection, 'q' to Quit.")
    
    while True:
        new_frame_time = time()
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
            
        imgOut = img.copy()

        try:
            # Run face detection
            results = face_model(img, conf=confidence_threshold, verbose=False)
            
            listBlur = []
            listInfo = []
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    
                    if conf > confidence_threshold:
                        # Apply offset
                        offsetW = (offsetPercentageW / 100) * w
                        x1, x2 = int(x1 - offsetW), int(x2 + offsetW)
                        offsetH = (offsetPercentageH / 100) * h
                        y1, y2 = int(y1 - offsetH), int(y2 + offsetH)
                        
                        # Ensure coordinates are within image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                        w, h = x2 - x1, y2 - y1
                        
                        # Skip if width or height is too small
                        if w <= 20 or h <= 20:
                            continue
                            
                        # Extract face
                        imgFace = img[y1:y2, x1:x2]
                        if imgFace.size == 0:  # Skip if face crop is empty
                            continue
                            
                        # Calculate blur
                        blurValue = int(calculate_blur(imgFace))
                        listBlur.append(blurValue > blurThreshold)
                        
                        # Determine class (real or fake)
                        if manual_class_id is not None:
                            current_class_id = manual_class_id
                        else:
                            # Auto-detect if real or fake (with error handling)
                            try:
                                current_class_id = 1 if is_real_face(imgFace, blurValue) else 0
                            except Exception as e:
                                print(f"Error in auto-detection: {e}")
                                current_class_id = 0  # Default to fake if there's an error
                        
                        # Format information for saving
                        ih, iw, _ = img.shape
                        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                        xcn, ycn = round(xc / iw, 6), round(yc / ih, 6)
                        wn, hn = round(w / iw, 6), round(h / ih, 6)
                        listInfo.append(f"{current_class_id} {xcn} {ycn} {wn} {hn}\n")
                        
                        # Draw on output image
                        label = classes[current_class_id]
                        color = (0, 255, 0) if current_class_id == 1 else (0, 0, 255)  # Green for real, Red for fake
                        cv2.rectangle(imgOut, (x1, y1), (x2, y2), color, 2)
                        cvzone.putTextRect(imgOut, f'{label} ({int(conf * 100)}%) Blur: {blurValue}', 
                                          (x1, y1 - 10), scale=0.8, thickness=2, colorR=color)
                
                # Save if not blurry and set to save
                if save and listBlur and all(listBlur):
                    timeNow = str(time()).replace(".", "")
                    os.makedirs(outputFolderPath, exist_ok=True)
                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                    with open(f"{outputFolderPath}/{timeNow}.txt", 'w') as f:
                        f.writelines(listInfo)
        
        except Exception as e:
            print(f"Error in processing frame: {e}")
        
        # Calculate and display FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(imgOut, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add info about current mode
        mode = "AUTO" if manual_class_id is None else classes[manual_class_id].upper()
        cv2.putText(imgOut, f"Mode: {mode}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Liveness Detection", imgOut)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            manual_class_id = 1
            print("Manual mode: REAL")
        elif key == ord('f'):
            manual_class_id = 0
            print("Manual mode: FAKE")
        elif key == ord('a'):
            manual_class_id = None
            print("Auto detection mode activated")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

