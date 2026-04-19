import cv2
import os
import numpy as np

def resize_with_padding(image_path, output_path, target_size=224):
    img = cv2.imread(image_path)
    if img is None: return False
    
    h, w, _ = img.shape
   
    
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    
    resized = cv2.resize(img, (new_w, new_h))
    
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    cv2.imwrite(output_path, canvas)
    return True


input_base = "classify"
output_base = "dataset_standardized"

for style in ["old_money", "minimalism"]:
    in_dir = os.path.join(input_base, style)
    out_dir = os.path.join(output_base, style)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    for filename in os.listdir(in_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            resize_with_padding(os.path.join(in_dir, filename), os.path.join(out_dir, filename))
            
            print(f"Standardized: {style}/{filename}")


print("\n--- All original images standardized successfully, aspect ratio preserved! ---")