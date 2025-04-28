import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def create_output_dir(base_dir):
    """
    Create output directory with timestamp
    
    Args:
        base_dir: Base directory
        
    Returns:
        Path to created output directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = os.path.join(base_dir, f"output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def draw_boxes(frame, objects, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on frame
    
    Args:
        frame: Input frame
        objects: List of objects with bounding boxes [x1, y1, x2, y2, id, class]
        color: Color to draw boxes
        thickness: Line thickness
        
    Returns:
        Frame with drawn bounding boxes
    """
    for obj in objects:
        x1, y1, x2, y2, track_id, class_id = obj
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw label
        label = f"ID:{track_id} Class:{class_id}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return frame

def plot_fps_graph(fps_list, output_path):
    """
    Plot FPS graph and save to file
    
    Args:
        fps_list: List of FPS values
        output_path: Path to save graph
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fps_list)
    plt.title('Processing Speed (FPS)')
    plt.xlabel('Frame')
    plt.ylabel('FPS')
    plt.grid(True)
    
    # Add average FPS line
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    plt.axhline(y=avg_fps, color='r', linestyle='-', label=f'Average: {avg_fps:.2f} FPS')
    
    plt.legend()
    plt.savefig(output_path)