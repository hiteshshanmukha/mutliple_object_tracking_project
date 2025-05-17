import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast, GradScaler

# MOT17 Dataset with Preprocessing
class MOT17Dataset(Dataset):
    def __init__(self, root_dir, split="train", max_size=800, use_gt=True):
        self.root_dir = root_dir
        self.split = split
        self.max_size = max_size  # Resize images to a maximum dimension
        self.use_gt = use_gt  # Use ground truth for training
        self.imgs = []
        self.annotations = []
        sequences = sorted(os.listdir(os.path.join(root_dir, split)))
        
        # Preload annotations to avoid repeated file I/O
        print(f"Loading {split} annotations...")
        for seq in sequences:
            if "FRCNN" not in seq:
                continue
            img_dir = os.path.join(root_dir, split, seq, "img1")
            
            # Choose between ground truth (for training) and detections (for testing)
            if use_gt and split == "train":
                ann_file = os.path.join(root_dir, split, seq, "gt/gt.txt")
            else:
                ann_file = os.path.join(root_dir, split, seq, "det/det.txt")
            
            img_files = sorted(os.listdir(img_dir))
            
            # Load sequence info
            seqinfo_path = os.path.join(root_dir, split, seq, "seqinfo.ini")
            seq_width, seq_height = 1920, 1080  # Default values
            if os.path.exists(seqinfo_path):
                with open(seqinfo_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "imWidth" in line:
                            seq_width = int(line.split("=")[1].strip())
                        elif "imHeight" in line:
                            seq_height = int(line.split("=")[1].strip())
            
            if os.path.exists(ann_file):
                with open(ann_file, "r") as f:
                    annotations = [line.strip().split(",") for line in f.readlines()]
                frame_dets = {}
                for ann in annotations:
                    frame_id = int(ann[0])
                    if frame_id not in frame_dets:
                        frame_dets[frame_id] = []
                    
                    if use_gt and split == "train":
                        # gt.txt format: frame_id, track_id, x, y, width, height, flag, class, visibility
                        # Convert to [x1, y1, x2, y2] format
                        x, y, w, h = float(ann[2]), float(ann[3]), float(ann[4]), float(ann[5])
                        if int(ann[6]) == 1:  # Only use objects marked as visible
                            box = [x, y, x + w, y + h]
                            frame_dets[frame_id].append({"box": box, "label": 1})  # Label 1 for pedestrian
                    else:
                        # det.txt format: frame_id, -1, x, y, width, height, confidence, -1, -1, -1
                        x, y, w, h = float(ann[2]), float(ann[3]), float(ann[4]), float(ann[5])
                        score = float(ann[6])
                        if score > 0.0:  # Filter low confidence detections
                            box = [x, y, x + w, y + h]
                            frame_dets[frame_id].append({"box": box, "label": 1, "score": score})
                
                for img_file in img_files:
                    frame_id = int(img_file.split(".")[0])
                    if frame_id in frame_dets:
                        self.imgs.append(os.path.join(img_dir, img_file))
                        self.annotations.append(frame_dets[frame_id])
            
        print(f"Loaded {len(self.imgs)} images for {split} split.")

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to a smaller size
        h, w = img.shape[:2]
        scale = self.max_size / max(h, w)
        if scale < 1:  # Only resize if image is larger than max_size
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        annotations = self.annotations[idx]
        boxes = [ann["box"] for ann in annotations]
        # Scale bounding boxes
        boxes = [[box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale] for box in boxes]
        labels = [ann["label"] for ann in annotations]
        
        img = F.to_tensor(img)
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([idx])}
        return img, target

# Faster R-CNN Model (Using Pre-trained Model from torchvision)
def get_faster_rcnn_model(num_classes=2, freeze_backbone=False):
    # Use pre-trained Faster R-CNN with ResNet-50 FPN backbone
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Freeze backbone if specified (optional)
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # Replace the classifier head for MOT17 (2 classes: background, pedestrian)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model

# Training Function with Mixed Precision
def train_model(model, train_loader, model_path="models/faster_rcnn_mot17.pt", num_epochs=1, device="cuda"):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    scaler = GradScaler()  # For mixed precision training
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            with autocast():  # Mixed precision
                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += losses.item()
        print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

# SORT Tracker (unchanged)
class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0],
                                             [0, 1, 0, 0, 0, 1, 0],
                                             [0, 0, 1, 0, 0, 0, 1],
                                             [0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.statePre = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0], np.float32)
        self.id = np.random.randint(10000)
        self.time_since_update = 0
        self.hits = 0
    
    def predict(self):
        pred = self.kf.predict()
        self.time_since_update += 1
        return pred[:4]
    
    def update(self, bbox):
        self.kf.correct(np.array(bbox, np.float32))
        self.time_since_update = 0
        self.hits += 1

class SORT:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def iou(self, bb1, bb2):
        x1, y1, x2, y2 = bb1
        x1_, y1_, x2_, y2_ = bb2
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        self.frame_count += 1
        trks = [t.predict() for t in self.trackers]
        matched, unmatched_dets, unmatched_trks = [], [], list(range(len(trks)))
        
        if len(trks) > 0 and len(detections) > 0:
            iou_matrix = np.array([[self.iou(det, trk) for trk in trks] for det in detections])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((r, c))
                    unmatched_trks.remove(c)
                else:
                    unmatched_dets.append(r)
            unmatched_dets = [i for i in range(len(detections)) if i not in [m[0] for m in matched]]
        
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))
        
        for i in reversed(unmatched_trks):
            self.trackers[i].time_since_update += 1
            if self.trackers[i].time_since_update > self.max_age:
                self.trackers.pop(i)
        
        for r, c in matched:
            self.trackers[c].update(detections[r])
        
        return [(t.id, t.predict()) for t in self.trackers if t.hits >= self.min_hits or self.frame_count <= self.min_hits]

# Tracking Function
def track_video(model_path, video_dir, output_path, device="cuda", conf_threshold=0.5):
    # Load the trained model
    model = get_faster_rcnn_model(num_classes=2)
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize tracker
    tracker = SORT(max_age=15, min_hits=2, iou_threshold=0.3)  # Adjusted parameters for better tracking
    imgs = sorted(os.listdir(video_dir))
    
    # Get video dimensions from first frame
    first_img = cv2.imread(os.path.join(video_dir, imgs[0]))
    h, w = first_img.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
    
    # Add counters for debugging
    total_frames = len(imgs)
    frames_with_detections = 0
    frames_with_tracks = 0
    
    print(f"Processing {len(imgs)} frames...")
    for img_file in tqdm(imgs):
        img_path = os.path.join(video_dir, img_file)
        img = cv2.imread(img_path)
        original_img = img.copy()  # Keep original for display
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
        
        # Debug: print raw detections
        print(f"Frame {img_file}: {len(boxes)} raw detections")
        
        # Scale boxes back to original image size
        if scale < 1:
            boxes = boxes / scale
            
        # Filter by confidence and class (only pedestrians)
        mask = (scores > conf_threshold) & (labels == 1)
        filtered_boxes = boxes[mask]
        
        # Debug: print filtered detections
        print(f"Frame {img_file}: {len(filtered_boxes)} filtered detections (conf > {conf_threshold})")
        
        if len(filtered_boxes) > 0:
            frames_with_detections += 1
        
        # Try a lower confidence threshold if no detections
        if len(filtered_boxes) == 0 and len(boxes) > 0:
            # Backup approach with lower threshold
            lower_threshold = 0.3
            mask = (scores > lower_threshold) & (labels == 1)
            filtered_boxes = boxes[mask]
            print(f"Trying lower threshold {lower_threshold}: {len(filtered_boxes)} detections")
        
        # Update tracker
        tracks = tracker.update(filtered_boxes)
        
        # Debug: print tracks
        print(f"Frame {img_file}: {len(tracks)} tracks")
        
        if len(tracks) > 0:
            frames_with_tracks += 1
        
        # Draw tracks on the original image
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            # Draw bounding box
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw ID
            cv2.putText(original_img, f"ID: {int(track_id)}", (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write frame to output video
        cap.write(original_img)
        
        # Save sample frames with detections for debugging
        if len(tracks) > 0 and frames_with_tracks <= 5:  # Save first 5 frames with tracks
            debug_path = os.path.join(os.path.dirname(output_path), f"debug_frame_{img_file}")
            cv2.imwrite(debug_path, original_img)
            print(f"Saved debug frame to {debug_path}")
    
    cap.release()
    print(f"Tracking completed. Output video saved to {output_path}")
    print(f"Summary: {frames_with_detections}/{total_frames} frames had detections")
    print(f"Summary: {frames_with_tracks}/{total_frames} frames had tracks")

# Main Execution
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model paths
    model_path = "/home/dc_gr1/Project/models/faster_rcnn_mot17.pt"
    
    # Training parameters
    train_model_flag = True
    num_epochs = 1  # Increased epochs for better training
    
    # Use pre-trained Faster R-CNN with ResNet-50 FPN
    model = get_faster_rcnn_model(num_classes=2, freeze_backbone=False)
    
    # Training
    if train_model_flag:
        print("Setting up training dataset...")
        train_dataset = MOT17Dataset(
            root_dir="/home/dc_gr1/Project/MOT17/MOT17", 
            split="train", 
            max_size=800,
            use_gt=True  # Use ground truth for training
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=2,  # Reduced batch size for memory constraints
            shuffle=True, 
            num_workers=4, 
            pin_memory=True, 
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        train_model(model, train_loader, model_path=model_path, num_epochs=num_epochs, device=device)
    
    # Tracking test sequence
    test_video_dir = "/home/dc_gr1/Project/MOT17/MOT17/test/MOT17-01-FRCNN/img1"
    output_video_path = "/home/dc_gr1/Project/output/tracked_mot17_01.mp4"
    
    print(f"Starting object tracking on test sequence...")
    
    # Add this before calling track_video
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} does not exist. Training may have failed.")
        print("Proceeding with a fresh model...")
        # Use a fresh pre-trained model without loading saved weights
        model = get_faster_rcnn_model(num_classes=2)
        track_video(None, test_video_dir, output_video_path, device=device, conf_threshold=0.3)
    else:
        track_video(model_path, test_video_dir, output_video_path, device=device, conf_threshold=0.3)