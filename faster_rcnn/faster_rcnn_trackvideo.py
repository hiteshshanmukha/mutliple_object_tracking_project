import torch
import torchvision
import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Faster R-CNN Model (Using Pre-trained Model from torchvision)
def get_faster_rcnn_model(num_classes=2):
    # Use pre-trained Faster R-CNN with ResNet-50 FPN backbone
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace the classifier head for MOT17 (2 classes: background, pedestrian)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model

# SORT Tracker implementation
import torch
import torchvision
import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import functional as F

# SORT Tracker fixed implementation
# SORT Tracker Implementation with Sequential IDs
class KalmanBoxTracker:
    count = 0  # Class variable for generating sequential IDs
    
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        """
        # Define constant velocity model
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0]], np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0], 
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1], 
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]], np.float32)
        
        # Initialize state with coordinates
        bbox_array = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0], dtype=np.float32)
        self.kf.statePre = bbox_array
        self.kf.statePost = bbox_array
        
        # Set measurement and process noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        
        # Generate sequential ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        # Track state variables
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        """
        self.age += 1
        self.time_since_update += 1
        prediction = self.kf.predict()
        return prediction[:4]
    
    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        self.hits += 1
        self.time_since_update = 0
        measurement = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
        self.kf.correct(measurement)

class SORT:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.25):
        """
        Initialize SORT parameters
        """
        self.max_age = max_age  # Maximum frames to keep track alive
        self.min_hits = min_hits  # Minimum hits before track is established
        self.iou_threshold = iou_threshold  # IOU threshold for matching
        self.trackers = []  # List of active trackers
        self.frame_count = 0  # Frame counter
        
        # Reset the KalmanBoxTracker counter when creating a new SORT instance
        KalmanBoxTracker.count = 0
    
    def iou(self, bb_test, bb_gt):
        """
        Compute IOU between two bounding boxes
        """
        # Ensure boxes have positive area
        if (bb_test[2] <= bb_test[0]) or (bb_test[3] <= bb_test[1]) or \
           (bb_gt[2] <= bb_gt[0]) or (bb_gt[3] <= bb_gt[1]):
            return 0.0
            
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])
        
        # Intersection area
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        intersection = w * h
        
        # Union area
        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_test + area_gt - intersection
        
        if union <= 0:
            return 0.0
            
        return intersection / union
    
    def update(self, dets):
        """
        Update tracker with new detections
        """
        self.frame_count += 1
        
        # Filter invalid detections
        valid_dets = []
        for det in dets:
            if (det[2] > det[0]) and (det[3] > det[1]):
                valid_dets.append(det)
        
        # Get predictions from existing trackers
        trks = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            # Check if prediction is valid
            if np.any(np.isnan(pos)):
                to_del.append(i)
            else:
                # Only keep valid predictions
                if pos[2] > pos[0] and pos[3] > pos[1]:
                    trks.append(pos)
                else:
                    to_del.append(i)
        
        # Remove invalid trackers
        for i in sorted(to_del, reverse=True):
            self.trackers.pop(i)
        
        # Match detections to trackers
        matched_indices = []
        if len(trks) > 0 and len(valid_dets) > 0:
            # Calculate IOU matrix
            iou_matrix = np.zeros((len(valid_dets), len(trks)), dtype=np.float32)
            for d, det in enumerate(valid_dets):
                for t, trk in enumerate(trks):
                    iou_matrix[d, t] = self.iou(det, trk)
            
            # Hungarian algorithm for assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = list(zip(row_ind, col_ind))
        
        # Match successful assignments
        matches = []
        unmatched_dets = list(range(len(valid_dets)))
        unmatched_trks = list(range(len(trks)))
        
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                # Remove from unmatched lists
                if m[0] in unmatched_dets:
                    unmatched_dets.remove(m[0])
                if m[1] in unmatched_trks:
                    unmatched_trks.remove(m[1])
                # Add to matches
                matches.append(m)
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(valid_dets[i])
            self.trackers.append(trk)
        
        # Update matched trackers
        for m in matches:
            self.trackers[m[1]].update(valid_dets[m[0]])
        
        # Remove dead trackers
        i = len(self.trackers) - 1
        while i >= 0:
            if self.trackers[i].time_since_update > self.max_age:
                self.trackers.pop(i)
            i -= 1
        
        # Return active tracks
        ret = []
        for trk in self.trackers:
            pos = trk.predict()
            if (pos[0] < pos[2]) and (pos[1] < pos[3]):  # Check valid box
                ret.append((trk.id, pos))
        
        return ret

# Updated tracking function that works directly with a video file
def track_video(model_path, input_video_path, output_path, device="cuda", conf_threshold=0.85):
    # Load the trained model
    model = get_faster_rcnn_model(num_classes=2)
    
    if model_path is None or not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} does not exist. Using default pre-trained model.")
    else:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize tracker with stricter parameters
    tracker = SORT(max_age=15, min_hits=5, iou_threshold=0.4)
    
    # Open the video file directly
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Add counters for debugging
    frames_with_detections = 0
    frames_with_tracks = 0
    frame_id = 0
    
    # NMS parameters
    nms_iou_threshold = 0.2
    
    print(f"Processing video with {frame_count} frames...")
    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break  # Break when video ends
            
            frame_id += 1
            pbar.update(1)
            
            # Optionally save individual frames (for debugging)
            # frame_filename = os.path.join("debug_frames", f"{frame_id:06d}.jpg")
            # os.makedirs("debug_frames", exist_ok=True)
            # cv2.imwrite(frame_filename, img)
            
            original_img = img.copy()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for model input
            h, w = img_rgb.shape[:2]
            max_size = 800
            scale = max_size / max(h, w)
            if scale < 1:
                new_h, new_w = int(h * scale), int(w * scale)
                input_img = cv2.resize(img_rgb, (new_w, new_h))
            else:
                input_img = img_rgb
                scale = 1.0
            
            img_tensor = F.to_tensor(input_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = model(img_tensor)[0]
                
            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            
            # Scale boxes back to original image size
            if scale < 1:
                boxes = boxes / scale
                
            # Filter by confidence and class (only pedestrians)
            mask = (scores > conf_threshold) & (labels == 1)
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            
            # Additional filtering based on box dimensions
            valid_indices = []
            for i, box in enumerate(filtered_boxes):
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                # Filter out boxes with unrealistic aspect ratios for pedestrians
                aspect_ratio = height / (width + 1e-6)
                
                # Stricter aspect ratio and minimum size requirements
                if 1.8 < aspect_ratio < 3.5 and height > 50 and width > 15:
                    valid_indices.append(i)
                    
            filtered_boxes = filtered_boxes[valid_indices]
            filtered_scores = filtered_scores[valid_indices]
            
            # Apply Non-Maximum Suppression to remove redundant boxes
            if len(filtered_boxes) > 0:
                # Convert to torch tensors for torchvision NMS function
                filtered_boxes_tensor = torch.from_numpy(filtered_boxes).float()
                filtered_scores_tensor = torch.from_numpy(filtered_scores).float()
                
                # Apply NMS and get indices of boxes to keep
                keep_indices = torchvision.ops.nms(
                    boxes=filtered_boxes_tensor,
                    scores=filtered_scores_tensor,
                    iou_threshold=nms_iou_threshold
                )
                
                # Keep only boxes that survived NMS
                filtered_boxes = filtered_boxes[keep_indices.numpy()]
                
                frames_with_detections += 1
            
            # Update tracker
            tracks = tracker.update(filtered_boxes)

            # Post-process tracks to remove overlaps
            if len(tracks) > 1:
                # Convert tracks to format for NMS
                track_boxes = np.array([bbox for _, bbox in tracks])
                track_ids = np.array([track_id for track_id, _ in tracks])
                
                # Calculate track scores based on tracker hit count 
                # (using a dummy score since we don't have real confidence scores for tracks)
                track_scores = np.ones(len(tracks))
                
                # Convert to torch tensors
                track_boxes_tensor = torch.from_numpy(track_boxes).float()
                track_scores_tensor = torch.from_numpy(track_scores).float()
                
                # Apply NMS to tracks
                keep_indices = torchvision.ops.nms(
                    boxes=track_boxes_tensor,
                    scores=track_scores_tensor,
                    iou_threshold=0.3  # Use a moderate threshold for tracks
                )
                
                # Filter tracks based on NMS results
                filtered_tracks = [(track_ids[i], track_boxes[i]) for i in keep_indices.numpy()]
                tracks = filtered_tracks

            # After the post-processing NMS for tracks but before drawing them
            if len(tracks) > 0:
                frames_with_tracks += 1  # Count frames with tracks ONLY ONCE
                
                # Handle occlusions - determine which boxes likely represent people in front
                # and hide boxes for people who are behind others
                visibility_mask = [True] * len(tracks)  # Initially assume all tracks are visible
                
                # Check each pair of boxes for significant overlap
                for i in range(len(tracks)):
                    if not visibility_mask[i]:
                        continue  # Skip if already hidden
                        
                    for j in range(i+1, len(tracks)):
                        if not visibility_mask[j]:
                            continue  # Skip if already hidden
                            
                        # Get bounding boxes
                        _, box_i = tracks[i]
                        _, box_j = tracks[j]
                        
                        # Calculate IoU between boxes
                        def calculate_iou(box1, box2):
                            # Convert to integers
                            box1 = [int(coord) for coord in box1]
                            box2 = [int(coord) for coord in box2]
                            
                            # Calculate intersection
                            x1 = max(box1[0], box2[0])
                            y1 = max(box1[1], box2[1])
                            x2 = min(box1[2], box2[2])
                            y2 = min(box1[3], box2[3])
                            
                            if x2 <= x1 or y2 <= y1:
                                return 0.0  # No overlap
                            
                            intersection = (x2 - x1) * (y2 - y1)
                            
                            # Calculate areas
                            area_i = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            area_j = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            
                            # Calculate IoU
                            iou = intersection / min(area_i, area_j)  # Use min area to be sensitive to partial overlaps
                            return iou
                        
                        iou = calculate_iou(box_i, box_j)
                        
                        # If boxes significantly overlap, determine which one is in front
                        if iou > 0.4:  # Threshold for significant overlap
                            # Assuming the box with the lower bottom edge (higher y2 coordinate) 
                            # is in front, as it's closer to the camera
                            if box_i[3] > box_j[3]:
                                # Box i is in front, hide box j
                                visibility_mask[j] = False
                            else:
                                # Box j is in front, hide box i
                                visibility_mask[i] = False
                                break  # Box i is hidden, no need to check against other boxes
                
                # Filter tracks based on visibility
                visible_tracks = [track for track, is_visible in zip(tracks, visibility_mask) if is_visible]
                
                # Replace the original tracks list with only the visible ones
                tracks = visible_tracks

            # Draw only the visible tracks on the original image
            for track_id, bbox in tracks:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Make sure coordinates are within image bounds
                x1 = max(0, min(x1, original_img.shape[1]-1))
                y1 = max(0, min(y1, original_img.shape[0]-1))
                x2 = max(0, min(x2, original_img.shape[1]-1))
                y2 = max(0, min(y2, original_img.shape[0]-1))
                
                # Only draw valid boxes
                if x2 > x1 and y2 > y1:
                    # Draw bounding box (thicker line for visibility)
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw ID with background for better visibility
                    text = f"ID: {int(track_id)}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(original_img, (x1, y1-25), (x1+text_size[0], y1), (0, 0, 0), -1)
                    cv2.putText(original_img, text, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(original_img)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Tracking completed. Output video saved to {output_path}")
    print(f"Summary: {frames_with_detections}/{frame_id} frames had detections")
    print(f"Summary: {frames_with_tracks}/{frame_id} frames had tracks")

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    model_path = "/home/dc_gr1/Project/10models/10epoch_rcnn_model_epoch10.pt"
    # Use direct video file path instead of directory
    input_video_path = "/home/dc_gr1/Project/MOT17-13-SDP-raw.mp4"
    output_video_path = "/home/dc_gr1/Project/output/filtered_track_output.mp4"
    
    # Run tracking on the video
    print(f"Starting object tracking on video: {input_video_path}...")
    
    # Check if model exists before trying to load it
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} does not exist. Using default pre-trained model.")
        track_video(None, input_video_path, output_video_path, device=device, conf_threshold=0.7)
    else:
        track_video(model_path, input_video_path, output_video_path, device=device, conf_threshold=0.7)