# Multiple Object Tracking Project

## Introduction

This project implements and compares multiple approaches to object detection and tracking with a focus on pedestrian tracking in the MOT17 dataset. The repository contains three different tracking implementations:

1. **Faster R-CNN with SORT Tracker** - Detection using Faster R-CNN with ResNet50 backbone and tracking via SORT (Simple Online and Realtime Tracking)
2. **YOLOv5 with Hungarian Algorithm** - YOLOv5 detector combined with Hungarian algorithm for track association
3. **YOLOv8 with DeepSORT** - YOLOv8 detector with DeepSORT tracker using appearance features and Kalman filtering

The project includes training scripts, evaluation metrics, and visualization tools to compare the performance of different approaches.

## Features

- Training object detection models on MOT17 dataset
- Multiple tracking algorithms implementation
- Comprehensive evaluation metrics (MOTA, MOTP, IDF1)
- Video visualization with bounding boxes and track IDs
- Performance benchmarking tools

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- OpenCV
- NumPy
- filterpy (for Kalman filtering)
- scipy
- matplotlib
- tqdm
- ultralytics (for YOLOv5/YOLOv8)
- motmetrics



## Dataset

The project uses the MOT17 dataset for training and evaluation. Download the dataset from the [MOTChallenge website](https://motchallenge.net/data/MOT17/).

Expected directory structure:
```
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── img1/
│   │   ├── gt/
│   │   └── seqinfo.ini
│   ├── MOT17-04-DPM/
│   │   ├── ...
│   └── ...
└── test/
    └── ...
```

## Usage

### 1. Faster R-CNN with SORT

#### Training:
```
python faster_rcnn_train.py --root_dir /path/to/MOT17 --output_dir ./models --epochs 10
```

#### Tracking:
```
python faster_rcnn_trackvideo.py --model_path ./models/faster_rcnn_mot17.pt --input_video path/to/video.mp4 --output_path ./results/tracked_video.mp4
```

#### Evaluation:
```
python fasterrcnn_metric.py --model_path ./models/faster_rcnn_mot17.pt --dataset_dir /path/to/MOT17 --sequence_name MOT17-02-DPM --output_dir ./results
```

### 2. YOLOv5 with Hungarian Algorithm

Run the Jupyter notebook MODEL3_yolov5_hungarian.ipynb which contains a comprehensive pipeline for training, tracking, and evaluation.

### 3. YOLOv8 with DeepSORT

Run the Jupyter notebook yolov8_Deepsort_scratch.ipynb for the complete pipeline.

## Evaluation Metrics

The project implements standard MOT metrics:

- **MOTA** (Multiple Object Tracking Accuracy): Overall tracking accuracy considering false positives, misses, and identity switches
- **MOTP** (Multiple Object Tracking Precision): Precision of object localization
- **IDF1** (ID F1 Score): F1 score for identity preservation
- **MT** (Mostly Tracked): Percentage of ground truth tracks tracked for at least 80% of their life span
- **ML** (Mostly Lost): Percentage of ground truth tracks tracked for less than 20% of their life span
- **FP** (False Positives): Number of false detections
- **FN** (False Negatives): Number of missed detections
- **IDSw** (ID Switches): Number of identity switches

## Results

The repository contains detailed evaluation results for each tracking approach. The metrics are calculated on several MOT17 sequences and can be visualized using the provided plotting functions.




## Acknowledgments

- MOT17 dataset from the MOTChallenge
- SORT tracker implementation based on the paper by Bewley et al.
- DeepSORT implementation inspired by the paper by Wojke et al.
- YOLOv5 and YOLOv8 from Ultralytics
