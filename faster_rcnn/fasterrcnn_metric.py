import os
import numpy as np
import pandas as pd
import cv2
import torch
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

class MOT17Metrics:
    """
    Evaluation metrics for Multiple Object Tracking on MOT17 dataset
    Implements CLEAR MOT metrics and additional tracking metrics
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters for metrics computation"""
        self.gt_tracks = {}  # {frame_id: {track_id: box}}
        self.pred_tracks = {}  # {frame_id: {track_id: box}}
        self.matches = {}  # {frame_id: [(gt_id, pred_id)]}
        
        # Accumulators for metrics
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        self.idsw = 0  # ID switches
        self.num_frames = 0
        self.gt_track_ids = set()
        self.pred_track_ids = set()
        
        # For tracking per GT object
        self.gt_track_map = defaultdict(lambda: {'total_frames': 0, 'matched_frames': 0, 'last_matched': None})
        
    def load_ground_truth(self, gt_file):
        """
        Load ground truth from MOT17 format file
        Each line: <frame_id>, <track_id>, <x>, <y>, <w>, <h>, <conf>, <class>, <visibility>
        """
        self.gt_tracks = defaultdict(dict)
        
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth file {gt_file} does not exist.")
            return
            
        df = pd.read_csv(gt_file, header=None)
        
        # Handle different GT file formats
        if len(df.columns) >= 7:
            # Standard MOT format
            if len(df.columns) == 9:
                df.columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
            else:
                df.columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf']
            
            # Filter for pedestrians (class=1) if class column exists
            if 'class' in df.columns:
                df = df[df['class'] == 1]
                
            # Process each detection
            for _, row in df.iterrows():
                frame_id = int(row['frame_id'])
                track_id = int(row['track_id'])
                
                # Convert x,y,w,h to x1,y1,x2,y2 (top-left, bottom-right)
                x, y, w, h = row['x'], row['y'], row['w'], row['h']
                box = [x, y, x + w, y + h]
                
                self.gt_tracks[frame_id][track_id] = box
                self.gt_track_ids.add(track_id)
                self.gt_track_map[track_id]['total_frames'] += 1
                
        self.num_frames = max(self.gt_tracks.keys()) if self.gt_tracks else 0
        print(f"Loaded {len(self.gt_tracks)} frames with {len(self.gt_track_ids)} ground truth tracks")
    
    def load_predictions(self, pred_file):
        """
        Load tracking predictions in same format as ground truth
        Each line: <frame_id>, <track_id>, <x>, <y>, <w>, <h>, <conf>, <-1>, <-1>, <-1>
        """
        self.pred_tracks = defaultdict(dict)
        
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} does not exist.")
            return
        
        # Check if file is empty
        if os.path.getsize(pred_file) == 0:
            print(f"Warning: Prediction file {pred_file} is empty.")
            return
        
        try:
            df = pd.read_csv(pred_file, header=None)
            
            if len(df) == 0:
                print(f"Warning: No data found in prediction file {pred_file}")
                return
                
            if len(df.columns) >= 6:
                # Handle different prediction file formats
                if len(df.columns) == 10:
                    df.columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'dummy1', 'dummy2', 'dummy3']
                else:
                    df.columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h']
                
                # Process each detection
                for _, row in df.iterrows():
                    frame_id = int(row['frame_id'])
                    track_id = int(row['track_id'])
                    
                    # Convert x,y,w,h to x1,y1,x2,y2 (top-left, bottom-right)
                    x, y, w, h = row['x'], row['y'], row['w'], row['h']
                    box = [x, y, x + w, y + h]
                    
                    self.pred_tracks[frame_id][track_id] = box
                    self.pred_track_ids.add(track_id)
                
                print(f"Loaded predictions for {len(self.pred_tracks)} frames with {len(self.pred_track_ids)} tracked objects")
            else:
                print(f"Warning: Unexpected format in prediction file {pred_file}")
        
        except Exception as e:
            print(f"Error loading predictions: {e}")
    
    def iou(self, box1, box2):
        """Calculate IOU between two boxes [x1, y1, x2, y2]"""
        # Find intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate areas
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IOU
        if union_area == 0:
            return 0
        return intersection_area / union_area
    
    def associate_detections(self, iou_threshold=0.5):
        """
        Associate ground truth and predicted detections using Hungarian algorithm
        Returns matches, unmatched_gt and unmatched_pred for each frame
        """
        self.matches = {}
        last_matches = {}  # To keep track of last matched IDs for ID switch computation
        
        for frame_id in range(1, self.num_frames + 1):
            if frame_id not in self.gt_tracks and frame_id not in self.pred_tracks:
                continue
                
            # Get ground truth and predictions for current frame
            gt_boxes = self.gt_tracks.get(frame_id, {})
            pred_boxes = self.pred_tracks.get(frame_id, {})
            
            # Calculate IOU matrix
            gt_ids = list(gt_boxes.keys())
            pred_ids = list(pred_boxes.keys())
            
            if not gt_ids or not pred_ids:
                # No GT or predictions in this frame
                self.fn += len(gt_ids)
                self.fp += len(pred_ids)
                self.matches[frame_id] = []
                continue
                
            iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            for i, gt_id in enumerate(gt_ids):
                for j, pred_id in enumerate(pred_ids):
                    iou_matrix[i, j] = self.iou(gt_boxes[gt_id], pred_boxes[pred_id])
            
            # Apply Hungarian algorithm for optimal assignment
            matched_indices = []
            if iou_matrix.size > 0:
                # Use linear_sum_assignment with negative IOUs to find max
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                for i, j in zip(row_ind, col_ind):
                    if iou_matrix[i, j] >= iou_threshold:
                        matched_indices.append((i, j))
            
            # Process matches, unmatched GTs and unmatched predictions
            matches = []
            matched_gt_indices = set()
            matched_pred_indices = set()
            
            for gt_idx, pred_idx in matched_indices:
                gt_id = gt_ids[gt_idx]
                pred_id = pred_ids[pred_idx]
                matches.append((gt_id, pred_id))
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(pred_idx)
                
                # Update tracking records for GT objects
                self.gt_track_map[gt_id]['matched_frames'] += 1
                
                # Check for ID switches
                if gt_id in last_matches and last_matches[gt_id] != pred_id:
                    self.idsw += 1
                
                # Update last match
                last_matches[gt_id] = pred_id
            
            # Count false positives and false negatives
            self.fn += len(gt_ids) - len(matched_gt_indices)
            self.fp += len(pred_ids) - len(matched_pred_indices)
            
            # Save matches for this frame
            self.matches[frame_id] = matches
    
    def compute_metrics(self, iou_threshold=0.5):
        """Compute CLEAR MOT and other tracking metrics"""
        # Associate detections first
        self.associate_detections(iou_threshold)
        
        # Calculate total ground truth objects
        total_gt = sum(len(boxes) for boxes in self.gt_tracks.values())
        
        # Calculate MOTA (Multiple Object Tracking Accuracy)
        mota = 1 - (self.fp + self.fn + self.idsw) / total_gt if total_gt > 0 else 0
        
        # Calculate MOTP (Multiple Object Tracking Precision)
        total_iou = 0
        total_matches = 0
        for frame_id, matches in self.matches.items():
            gt_boxes = self.gt_tracks.get(frame_id, {})
            pred_boxes = self.pred_tracks.get(frame_id, {})
            
            for gt_id, pred_id in matches:
                iou_val = self.iou(gt_boxes[gt_id], pred_boxes[pred_id])
                total_iou += iou_val
                total_matches += 1
                
        motp = total_iou / total_matches if total_matches > 0 else 0
        
        # Calculate MT, PT, ML (Mostly tracked, partially tracked, mostly lost)
        gt_track_stats = {
            'MT': 0,  # Mostly tracked: tracked for at least 80% of lifespan
            'PT': 0,  # Partially tracked: tracked between 20% and 80%
            'ML': 0   # Mostly lost: tracked for less than 20% of lifespan
        }
        
        for track_id, stats in self.gt_track_map.items():
            if stats['total_frames'] == 0:
                continue
                
            tracked_ratio = stats['matched_frames'] / stats['total_frames']
            
            if tracked_ratio >= 0.8:
                gt_track_stats['MT'] += 1
            elif tracked_ratio < 0.2:
                gt_track_stats['ML'] += 1
            else:
                gt_track_stats['PT'] += 1
                
        # Calculate relative metrics
        mt_ratio = gt_track_stats['MT'] / len(self.gt_track_ids) if self.gt_track_ids else 0
        ml_ratio = gt_track_stats['ML'] / len(self.gt_track_ids) if self.gt_track_ids else 0
        pt_ratio = gt_track_stats['PT'] / len(self.gt_track_ids) if self.gt_track_ids else 0
        
        # Prepare results dictionary
        results = {
            'MOTA': mota,
            'MOTP': motp,
            'FP': self.fp,
            'FN': self.fn,
            'IDSw': self.idsw,
            'MT': gt_track_stats['MT'],
            'PT': gt_track_stats['PT'],
            'ML': gt_track_stats['ML'],
            'MT_Ratio': mt_ratio,
            'ML_Ratio': ml_ratio,
            'PT_Ratio': pt_ratio,
            'GT_Tracks': len(self.gt_track_ids),
            'Pred_Tracks': len(self.pred_track_ids)
        }
        
        return results

def save_tracking_results(tracks_by_frame, output_file):
    """
    Save tracking results in MOT17 format
    Format: <frame_id>,<track_id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1
    """
    with open(output_file, 'w') as f:
        for frame_id, tracks in tracks_by_frame.items():
            for track_id, box in tracks.items():
                # Convert [x1, y1, x2, y2] to [x, y, w, h]
                x = box[0]
                y = box[1]
                w = box[2] - box[0]
                h = box[3] - box[1]
                # Write in MOT17 format (frame_id, track_id, x, y, w, h, confidence, -1, -1, -1)
                f.write(f"{frame_id},{track_id},{x},{y},{w},{h},1,-1,-1,-1\n")

def run_tracker_and_evaluate(model_path, dataset_path, sequence_name, output_dir):
    """Run tracker on a sequence and evaluate its performance"""
    import sys
    import importlib.util
    import numpy as np
    import torch
    import cv2
    from track_old import get_faster_rcnn_model, SORT
    
    # Setup paths
    img_dir = os.path.join(dataset_path, "img1")
    gt_file = os.path.join(dataset_path, "gt/gt.txt")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, f"{sequence_name}_tracked.mp4")
    output_txt = os.path.join(output_dir, f"{sequence_name}_results.txt")
    
    # Load model
    model = get_faster_rcnn_model(num_classes=2)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Get all image files
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if not img_files:
        print(f"Error: No images found in {img_dir}")
        return None
    
    # Initialize tracker and storage for results
    tracker = SORT(max_age=30, min_hits=3, iou_threshold=0.3)
    tracks_by_frame = {}
    
    # Initialize video writer
    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    h, w, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 30, (w, h))
    
    # Count frames with detections and tracks
    frames_with_detections = 0
    frames_with_tracks = 0
    
    # Process each frame
    for frame_idx, img_file in enumerate(tqdm(img_files)):
        # Get frame number from filename
        frame_id = int(img_file.split('.')[0])
        
        # Load and process image
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to(device)
        
        # Get detections
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # Extract detections
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter detections (pedestrians only with score > 0.5)
        mask = (scores > 0.5) & (labels == 1)
        filtered_boxes = boxes[mask]
        
        if len(filtered_boxes) > 0:
            frames_with_detections += 1
        
        # Update tracker 
        tracks = tracker.update(filtered_boxes)
        
        if len(tracks) > 0:
            frames_with_tracks += 1
        
        # Store tracks for this frame
        frame_tracks = {}
        
        # Visualize and store tracks
        for d in tracks:
            # Handle the actual format returned by your SORT implementation
            if isinstance(d, tuple) and len(d) == 2:
                # Format is (track_id, bbox_array)
                track_id = int(d[0])
                bbox = d[1]
            elif len(d) >= 5:
                # Standard SORT format: [x1, y1, x2, y2, track_id]
                track_id = int(d[4])
                bbox = d[:4]
            else:
                # Unknown format
                print(f"Warning: Track with unexpected format detected: {d}")
                continue
            
            frame_tracks[track_id] = bbox
            
            # Draw on image
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save tracks for this frame
        tracks_by_frame[frame_id] = frame_tracks
        
        # Write frame to video
        video_writer.write(img)
    
    # Release video writer
    video_writer.release()
    print(f"Tracking completed. Output video saved to {output_video}")
    print(f"Summary: {frames_with_detections}/{len(img_files)} frames had detections")
    print(f"Summary: {frames_with_tracks}/{len(img_files)} frames had tracks")
    
    # Save tracking results to text file
    save_tracking_results(tracks_by_frame, output_txt)
    print(f"Tracking results saved to {output_txt}")
    
    # Evaluate metrics
    metrics = MOT17Metrics()
    metrics.load_ground_truth(gt_file)
    metrics.load_predictions(output_txt)
    results = metrics.compute_metrics()
    
    return results

def generate_report(results, output_file=None):
    """Generate a formatted report of tracking metrics"""
    if not results:
        print("No results to report.")
        return
        
    report = "\n" + "="*50 + "\n"
    report += "MOT17 TRACKING EVALUATION RESULTS\n"
    report += "="*50 + "\n\n"
    
    # Main metrics
    report += f"MOTA:  {results['MOTA']:.4f} (Multiple Object Tracking Accuracy)\n"
    report += f"MOTP:  {results['MOTP']:.4f} (Multiple Object Tracking Precision)\n"
    report += f"IDSw:  {results['IDSw']} (ID Switches)\n\n"
    
    # Detection metrics
    report += "Detection metrics:\n"
    report += f"  FP:    {results['FP']} (False Positives)\n"
    report += f"  FN:    {results['FN']} (False Negatives)\n\n"
    
    # Track metrics
    report += "Track metrics:\n"
    report += f"  MT:    {results['MT']} ({results['MT_Ratio']:.2%}) (Mostly Tracked)\n"
    report += f"  PT:    {results['PT']} ({results['PT_Ratio']:.2%}) (Partially Tracked)\n"
    report += f"  ML:    {results['ML']} ({results['ML_Ratio']:.2%}) (Mostly Lost)\n\n"
    
    # Track statistics
    report += f"Ground truth tracks: {results['GT_Tracks']}\n"
    report += f"Predicted tracks:    {results['Pred_Tracks']}\n"
    
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report

def evaluate_all_sequences(model_path, dataset_dir, output_dir):
    """Evaluate tracker on all MOT17 sequences"""
    # Find all sequence directories
    train_dir = os.path.join(dataset_dir, "train")
    sequences = []
    
    for seq_name in os.listdir(train_dir):
        if os.path.isdir(os.path.join(train_dir, seq_name)):
            sequences.append(seq_name)
    
    results = {}
    for seq in sequences:
        print(f"\nEvaluating sequence: {seq}")
        seq_results = run_tracker_and_evaluate(
            model_path=model_path,
            dataset_path=os.path.join(train_dir, seq),
            sequence_name=seq,
            output_dir=os.path.join(output_dir, seq)
        )
        
        if seq_results:
            results[seq] = seq_results
            generate_report(seq_results, os.path.join(output_dir, f"{seq}_report.txt"))
    
    # Calculate average results across all sequences
    if results:
        avg_results = {
            'MOTA': sum(r['MOTA'] for r in results.values()) / len(results),
            'MOTP': sum(r['MOTP'] for r in results.values()) / len(results),
            'FP': sum(r['FP'] for r in results.values()),
            'FN': sum(r['FN'] for r in results.values()),
            'IDSw': sum(r['IDSw'] for r in results.values()),
            'MT': sum(r['MT'] for r in results.values()),
            'PT': sum(r['PT'] for r in results.values()),
            'ML': sum(r['ML'] for r in results.values()),
            'GT_Tracks': sum(r['GT_Tracks'] for r in results.values()),
            'Pred_Tracks': sum(r['Pred_Tracks'] for r in results.values()),
            'MT_Ratio': sum(r['MT_Ratio'] for r in results.values()) / len(results),
            'ML_Ratio': sum(r['ML_Ratio'] for r in results.values()) / len(results),
            'PT_Ratio': sum(r['PT_Ratio'] for r in results.values()) / len(results)
            }
        
        
        # Print average results
        print("\nAverage Results Across All Sequences:")
        generate_report(avg_results, os.path.join(output_dir, "average_results_report.txt"))
        
        return results, avg_results

def main():
    """
    Main function to evaluate tracking performance on MOT17 dataset
    Uses hardcoded paths instead of command-line arguments
    """
    # Hardcoded paths
    model_path = "/home/dc_gr1/Project/10models/10epoch_rcnn_model_epoch10.pt"
    dataset_dir = "/home/dc_gr1/Project/MOT17/MOT17"
    output_dir = "/home/dc_gr1/Project/motmetrics"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("MOT17 Tracking Metrics Evaluation")
    print(f"Model path: {model_path}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    
    # Uncomment to evaluate a specific sequence
    sequence_name = "MOT17-10-SDP"
    print(f"\nEvaluating single sequence: {sequence_name}")
    results = run_tracker_and_evaluate(
        model_path=model_path,
        dataset_path=os.path.join(dataset_dir, "train", sequence_name),
        sequence_name=sequence_name,
        output_dir=os.path.join(output_dir, sequence_name)
    )
    if results:
        generate_report(results, os.path.join(output_dir, f"{sequence_name}_report.txt"))
    
    # Evaluate all sequences
    print("\nEvaluating all sequences...")
    results, avg_results = evaluate_all_sequences(
        model_path=model_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir
    )
    
    print("\nEvaluation complete! Results saved to:", output_dir)

if __name__ == "__main__":
    main()