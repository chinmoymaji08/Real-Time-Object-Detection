import argparse
import cv2
import numpy as np
import time
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from utils import draw_boxes, create_output_dir, plot_fps_graph
from types import SimpleNamespace

class ObjectChangeDetector:
    def __init__(self, model_path='yolov8n.pt', conf_thres=0.5, iou_thres=0.45, device='cuda'):
        """
        Initialize the object change detector with the specified model and parameters
        
        Args:
            model_path: Path to the YOLOv8 model
            conf_thres: Confidence threshold for detections
            iou_thres: IOU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize ByteTracker
        # Inside ObjectChangeDetector __init__
        tracker_config = SimpleNamespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30,
            mot20=False
        )
        self.tracker = BYTETracker(tracker_config)
        
        # Initialize object history
        self.objects = {}  # track_id -> {class_id, last_seen, bbox, consecutive_absent}
        self.object_history = defaultdict(list)  # frame_id -> [objects]
        self.stationary_threshold = 0.1  # IOU threshold to consider an object stationary
        self.missing_threshold = 15  # Number of frames before considering an object missing
        self.appearance_threshold = 10  # Number of frames before considering a new object
        self.frame_count = 0
        
        # Colors for visualization
        self.colors = {
            'detected': (0, 255, 0),  # Green for detected objects
            'missing': (0, 0, 255),   # Red for missing objects
            'new': (255, 0, 0)        # Blue for new objects
        }

    def detect_objects(self, frame):
        """
        Detect objects in the given frame using YOLOv8
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detections [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model(frame, verbose=False, conf=self.conf_thres, iou=self.iou_thres)
        detections = []
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                detections.append([x1, y1, x2, y2, conf, cls])
        
        return detections

    def update_tracking(self, frame, detections):
        """
        Update object tracking with new detections
        
        Args:
            frame: Current video frame
            detections: List of detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            List of tracked objects
        """
        if not detections:
            return []
            
        # Convert detections to format expected by ByteTracker
        tracker_dets = np.array(detections)

        img_info = (frame.shape[0], frame.shape[1])  # height, width

        # Run tracker
        online_targets = self.tracker.update(
            tracker_dets[:, :5],  # bbox + conf
            img_info,
            img_info
        )

        # Convert tracker output to our format
        tracked_objects = []
        
        for t in online_targets:
            track_id = t.track_id
            tlwh = t.tlwh
            x1, y1 = tlwh[0], tlwh[1]
            x2, y2 = x1 + tlwh[2], y1 + tlwh[3]
            
            # DO NOT ACCESS t.cls or t.class_id
            # 🟰 Set class_id manually to 0 (or unknown)
            class_id = 0
            
            tracked_objects.append([x1, y1, x2, y2, track_id, class_id])
        
        return tracked_objects

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou

    def update_object_states(self, tracked_objects):
        """
        Update the state of all tracked objects and detect missing/new objects
        
        Args:
            tracked_objects: List of tracked objects [x1, y1, x2, y2, track_id, class_id]
            
        Returns:
            Tuple of (detected_objects, missing_objects, new_objects)
        """
        current_frame_ids = set()
        detected_objects = []
        missing_objects = []
        new_objects = []
        
        # Update tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, class_id = obj
            bbox = [x1, y1, x2, y2]
            current_frame_ids.add(track_id)
            
            if track_id in self.objects:
                # Existing object
                self.objects[track_id].update({
                    'class_id': class_id,
                    'bbox': bbox,
                    'last_seen': self.frame_count,
                    'consecutive_absent': 0
                })
                detected_objects.append(obj)
            else:
                # New object candidate
                self.objects[track_id] = {
                    'class_id': class_id,
                    'bbox': bbox,
                    'last_seen': self.frame_count,
                    'consecutive_absent': 0,
                    'first_seen': self.frame_count,
                    'confirmed': False
                }
                
                # Check if it's genuinely new
                if self.frame_count - self.objects[track_id]['first_seen'] > self.appearance_threshold:
                    self.objects[track_id]['confirmed'] = True
                    new_objects.append(obj)
                else:
                    detected_objects.append(obj)
        
        # Check for missing objects
        for track_id, obj_info in list(self.objects.items()):
            if track_id not in current_frame_ids:
                obj_info['consecutive_absent'] += 1
                
                # If object has been absent for a threshold number of frames, mark as missing
                if obj_info['consecutive_absent'] >= self.missing_threshold and obj_info.get('confirmed', True):
                    missing_obj = obj_info['bbox'] + [track_id, obj_info['class_id']]
                    missing_objects.append(missing_obj)
                
                # Remove objects that have been absent for too long
                if obj_info['consecutive_absent'] > self.missing_threshold * 2:
                    del self.objects[track_id]
        
        # Save object history for this frame
        self.object_history[self.frame_count] = {
            'detected': detected_objects,
            'missing': missing_objects,
            'new': new_objects
        }
        
        return detected_objects, missing_objects, new_objects

    def process_frame(self, frame):
        """
        Process a single frame, detecting objects and changes
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with detected changes
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Update object tracking
        tracked_objects = self.update_tracking(frame, detections)
        
        # Update object states and detect changes
        detected_objects, missing_objects, new_objects = self.update_object_states(tracked_objects)
        
        # Visualize results
        annotated_frame = frame.copy()
        
        # Draw detected objects
        for obj in detected_objects:
            x1, y1, x2, y2, track_id, class_id = obj
            label = f"ID:{track_id} Class:{class_id}"
            color = self.colors['detected']
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw missing objects
        for obj in missing_objects:
            x1, y1, x2, y2, track_id, class_id = obj
            label = f"MISSING ID:{track_id} Class:{class_id}"
            color = self.colors['missing']
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw new objects
        for obj in new_objects:
            x1, y1, x2, y2, track_id, class_id = obj
            label = f"NEW ID:{track_id} Class:{class_id}"
            color = self.colors['new']
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, len(detected_objects), len(missing_objects), len(new_objects)

def process_video(input_video_path, output_dir, model_path='yolov8n.pt', conf_thres=0.5, device='cuda'):
    """
    Process a video file, detecting object changes
    
    Args:
        input_video_path: Path to input video file
        output_dir: Directory to save output
        model_path: Path to YOLOv8 model
        conf_thres: Confidence threshold
        device: Device to run inference on
    """
    # Initialize detector
    detector = ObjectChangeDetector(model_path=model_path, conf_thres=conf_thres, device=device)
    
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    output_path = os.path.join(output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    total_time = 0
    fps_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timer
        start_time = time.time()
        
        # Process frame
        annotated_frame, detected_count, missing_count, new_count = detector.process_frame(frame)
        
        # Calculate processing time
        end_time = time.time()
        process_time = end_time - start_time
        total_time += process_time
        
        # Calculate FPS
        current_fps = 1 / process_time if process_time > 0 else 0
        fps_list.append(current_fps)
        
        # Add FPS info to frame
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Objects: {detected_count} | Missing: {missing_count} | New: {new_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Display progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames | Current FPS: {current_fps:.2f}")
    
    # Calculate average FPS
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    
    # Create FPS graph
    fps_graph_path = os.path.join(output_dir, 'fps_graph.png')
    plot_fps_graph(fps_list, fps_graph_path)
    
    # Release resources
    cap.release()
    out.release()
    
    # Save performance metrics
    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        f.write(f"Total frames: {frame_count}\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n")
        f.write(f"Min FPS: {min(fps_list):.2f}\n")
        f.write(f"Max FPS: {max(fps_list):.2f}\n")
    
    return frame_count, total_time, avg_fps

def main():
    """Main function to run the object change detection pipeline"""
    parser = argparse.ArgumentParser(description="Real-time Detection of Object Missing and New Object Placement in Video")
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    
    # Process video
    process_video(args.input, output_dir, args.model, args.conf, args.device)

if __name__ == "__main__":
    main()