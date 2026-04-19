import cv2
import numpy as np
import tensorflow as tf
import os
from ultralytics import YOLO


model_path = 'identity_model.keras'
if not os.path.exists(model_path):
    print("Error: Cannot find identity_model.keras")
    exit()

print("Loading the AI Brain (Keras)...")
model = tf.keras.models.load_model(model_path)
class_names = ['minimalism', 'old_Money'] 

print("Loading the New Eyes (YOLOv8)...")
yolo_model = YOLO('yolov8n.pt') 


def analyze_clothing_part(crop_img):
    if crop_img.size == 0: 
        return 0.0, 0.0, (128, 128, 128) 
    
    target_h, target_w = crop_img.shape[:2]
    scale = 224 / max(target_h, target_w)
    new_h, new_w = int(target_h * scale), int(target_w * scale)
    resized = cv2.resize(crop_img, (new_w, new_h))
    
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    y_offset = (224 - new_h) // 2
    x_offset = (224 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    img_array = tf.expand_dims(canvas, 0)
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0]) 
    
    conf_min = 100 * score[0] 
    conf_old = 100 * score[1] 
    
    winner_label = class_names[np.argmax(score)]
    box_color = (0, 255, 0) if winner_label == 'old_Money' else (255, 255, 0)
    
    return conf_min, conf_old, box_color



cap = cv2.VideoCapture(1)

print("Silent Algorithmic Gaze Activated! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h_img, w_img, _ = frame.shape

    
    results = yolo_model(frame, classes=[0], verbose=False) 
    boxes = results[0].boxes.xyxy.cpu().numpy() 

    
    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            box_h = y2 - y1
            
            
            neck_y = y1 + int(box_h * 0.15)  
            waist_y = y1 + int(box_h * 0.50) 
            feet_y = y2                      

            
            neck_y, waist_y, feet_y = max(0, neck_y), min(h_img, waist_y), min(h_img, feet_y)
            x1, x2 = max(0, x1), min(w_img, x2)

            
            top_crop = frame[neck_y:waist_y, x1:x2]
            top_min, top_old, top_color = analyze_clothing_part(top_crop)

            
            bottom_crop = frame[waist_y:feet_y, x1:x2]
            bot_min, bot_old, bot_color = analyze_clothing_part(bottom_crop)

            
            cv2.rectangle(frame, (x1, neck_y), (x2, waist_y), top_color, 2)
            cv2.putText(frame, f"[TOP] MINIMALISM: {top_min:.1f}%", (x1, max(30, neck_y + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"[TOP] OLD MONEY: {top_old:.1f}%", (x1, max(30, neck_y + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.rectangle(frame, (x1, waist_y), (x2, feet_y), bot_color, 2)
            cv2.putText(frame, f"[BOT] MINIMALISM: {bot_min:.1f}%", (x1, max(30, waist_y + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"[BOT] OLD MONEY: {bot_old:.1f}%", (x1, max(30, waist_y + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow('Static Identity - Algorithmic Gaze', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)