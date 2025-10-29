import tritonclient.http as httpclient
import numpy as np
import cv2

class TritonYOLOv11Client:
    def __init__(self, url="localhost:8000", model_name="yolov11m"):
        self.triton_client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name
        
    def preprocess(self, image_path):
        """Preprocess image for YOLOv11"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_shape = img.shape[:2]
        
        # Resize to 640x640
        img_resized = cv2.resize(img, (640, 640))
        
        # Normalize and transpose
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, orig_shape
    
    def infer(self, image_path):
        """Run inference on Triton"""
        # Preprocess
        input_data, orig_shape = self.preprocess(image_path)
        
        # Setup inputs
        inputs = []
        inputs.append(httpclient.InferInput('images', input_data.shape, "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Setup outputs
        outputs = []
        outputs.append(httpclient.InferRequestedOutput('output0'))
        
        # Run inference
        response = self.triton_client.infer(
            self.model_name,
            inputs,
            outputs=outputs
        )
        
        # Get results
        output = response.as_numpy('output0')
        print(f"Output shape from server: {output.shape}")
        return output, orig_shape
    
    def postprocess(self, output, orig_shape, conf_threshold=0.25):
        """Post-process YOLOv11 output with shape [1, 5, 8400]"""
        if output is None:
            return []
        
        # Output shape: [1, 5, 8400]
        output = output[0]  # Remove batch: [5, 8400]
        output = output.T   # Transpose to [8400, 5]
        
        boxes = []
        scores = []
        
        for detection in output:
            x, y, w, h, confidence = detection
            
            if confidence >= conf_threshold:
                # Convert to xyxy format and scale to original image
                x1 = int((x - w/2) * orig_shape[1] / 640)
                y1 = int((y - h/2) * orig_shape[0] / 640)
                x2 = int((x + w/2) * orig_shape[1] / 640)
                y2 = int((y + h/2) * orig_shape[0] / 640)
                
                # Clip to image boundaries
                x1 = max(0, min(x1, orig_shape[1]))
                y1 = max(0, min(y1, orig_shape[0]))
                x2 = max(0, min(x2, orig_shape[1]))
                y2 = max(0, min(y2, orig_shape[0]))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
        
        # Apply NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
            final_boxes = []
            
            if len(indices) > 0:
                for i in indices.flatten():
                    final_boxes.append({
                        'bbox': boxes[i],
                        'score': scores[i]
                    })
            return final_boxes
        
        return []

# Usage
if __name__ == "__main__":
    # Initialize client
    client = TritonYOLOv11Client(url="localhost:8000")
    
    # Path to your test image (UPDATE THIS!)
    image_path = "/home/testing/mobile_ocr/1000218262.jpeg"
    
    print(f"Running inference on {image_path}...")
    
    try:
        # Run inference
        output, orig_shape = client.infer(image_path)
        
        # Post-process results
        detections = client.postprocess(output, orig_shape, conf_threshold=0.3)
        
        # Print results
        print(f"\nFound {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"{i+1}. Confidence: {det['score']:.3f}, BBox: {det['bbox']}")
            
    except Exception as e:
        print(f"Error: {e}")