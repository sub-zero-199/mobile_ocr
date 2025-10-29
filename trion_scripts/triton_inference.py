#!/usr/bin/env python3
import tritonclient.http as httpclient
import numpy as np
import cv2


def postprocess_yolo_best_only(output, conf_threshold=0.8, img_width=640, img_height=640):
    """
    Post-process YOLOv11 output and return ONLY the highest confidence detection
    Output format: [x_center, y_center, width, height, confidence]
    """
    detections = output[0].T  # Shape: [8400, 5]
    
    best_box = None
    best_confidence = 0.0
    
    for detection in detections:
        x_center, y_center, width, height, confidence = detection
        
        if confidence > conf_threshold and confidence > best_confidence:
            # Check if values are normalized (0-1 range) or already in pixels
            if x_center <= 1.0 and y_center <= 1.0:
                # Values are normalized, convert to pixels
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
            else:
                # Already in pixel coordinates
                x_center_px = x_center
                y_center_px = y_center
                width_px = width
                height_px = height
            
            # Convert center coordinates to corner coordinates
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            # Clip to image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            best_box = [x1, y1, x2, y2]
            best_confidence = float(confidence)
    
    return best_box, best_confidence


def run_triton_inference(image_path, model_name="yolov11m", server_url="localhost:8000"):
    """
    Run inference on Triton Server and return the best detection only
    """
    # 1. Setup connection to Triton Server
    triton_client = httpclient.InferenceServerClient(url=server_url, verbose=False)
    
    # 2. Check if model is ready
    if triton_client.is_model_ready(model_name):
        print(f"Model '{model_name}' is ready!")
    else:
        raise Exception(f"Model '{model_name}' is NOT ready")
    
    # 3. Load and preprocess image
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    
    # Normalize and convert to NCHW format: [1, 3, 640, 640]
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
    
    # 4. Create input object
    inputs = httpclient.InferInput("images", img_input.shape, datatype="FP32")
    inputs.set_data_from_numpy(img_input, binary_data=True)
    
    # 5. Create output object
    outputs = httpclient.InferRequestedOutput("output0", binary_data=True)
    
    # 6. Send inference request
    print(f"Sending inference request for {image_path}...")
    results = triton_client.infer(model_name=model_name, inputs=[inputs], outputs=[outputs])
    
    # 7. Get raw inference results
    output_data = results.as_numpy("output0")
    print(f"Raw output shape: {output_data.shape}")
    
    # 8. Get only the best detection
    best_box, best_confidence = postprocess_yolo_best_only(
        output_data, conf_threshold=0.5, img_width=640, img_height=640
    )
    
    if best_box is None:
        print("No detection found above confidence threshold!")
        return None, 0.0, img
    
    print(f"Best detection confidence: {best_confidence:.4f}")
    
    # 9. Scale box back to original image size
    x1, y1, x2, y2 = best_box
    x1_scaled = int(x1 * original_width / 640)
    y1_scaled = int(y1 * original_height / 640)
    x2_scaled = int(x2 * original_width / 640)
    y2_scaled = int(y2 * original_height / 640)
    
    best_box_scaled = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
    
    return best_box_scaled, best_confidence, img


def visualize_result(img, box, confidence):
    """
    Draw the single best detection on the image
    """
    if box is None:
        return img
    
    img_vis = img.copy()
    x1, y1, x2, y2 = box
    
    # Draw rectangle
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Add confidence label
    label = f"License Plate: {confidence:.2f}"
    cv2.putText(img_vis, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return img_vis


if __name__ == "__main__":
    import os
    from datetime import datetime
    
    # Path to your test image
    image_path = "/home/testing/mobile_ocr/1000218262.jpeg"
    
    # Run inference
    best_box, best_confidence, original_img = run_triton_inference(
        image_path=image_path,
        model_name="yolov11m",
        server_url="localhost:8000"
    )
    
    # Print result
    print("\n=== DETECTION RESULT ===")
    if best_box:
        print(f"License Plate Box: {best_box}")
        print(f"Confidence: {best_confidence:.4f}")
        
        # Create output directory if it doesn't exist
        output_dir = "/home/testing/mobile_ocr/trioton_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/result_{timestamp}.jpg"
        
        # Visualize and save result
        img_with_box = visualize_result(original_img, best_box, best_confidence)
        cv2.imwrite(output_path, img_with_box)
        print(f"\nVisualization saved to: {output_path}")
    else:
        print("No license plate detected!")
