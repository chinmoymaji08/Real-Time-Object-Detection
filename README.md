# Real-Time Detection of Missing and New Object Placement in Video

## 🚀 Project Overview

This project implements a **real-time video analytics pipeline** capable of:
- Detecting when objects **go missing** from a scene,
- Detecting when **new objects** are placed in the scene.

Despite being optimized for **resource-constrained hardware**, the system achieves stable performance with an average of **6.11 FPS** on an Intel Core i3 laptop with integrated graphics.

---

## 🛠 System Architecture

The pipeline consists of four main modules:

1. **Object Detection**:  
   - Model: [YOLOv8n (Nano)](https://github.com/ultralytics/ultralytics)
   - Lightweight and fast object detection.

2. **Object Tracking**:  
   - Tracker: [ByteTrack](https://github.com/ifzhang/ByteTrack)
   - Robust multi-object tracking through occlusions.

3. **Scene State Management**:  
   - Maintains object history,
   - Detects missing and newly appearing objects.

4. **Visualization Engine**:  
   - Annotates frames with bounding boxes,
   - Shows live FPS and event logs,
   - Saves output video and FPS graphs.

---

## 🔥 Object Detection and Tracking Flow

```mermaid
flowchart LR
    A[Video Frame] --> B[YOLOv8n Object Detection]
    B --> C[ByteTrack Tracking]
    C --> D[Scene State Management]
    D --> E[Change Detection (Missing/New)]
    E --> F[Visualization & Output]```

