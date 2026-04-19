# Static Identity

An interactive camera installation that critiques the algorithmic gaze by quantifying viewers into rigid, commodified fashion labels (e.g., 'Old Money', 'Minimalism') using YOLOv8 and a custom-trained CNN.

## Prerequisites
Before running the project, ensure you have the following hardware and software:
* A working Webcam (Built-in or USB).
* **Python 3.8 or higher** installed on your machine.

## Installation

**1. Clone the repository or download the files:**
Ensure all files (`live_identity.py`, `identity_model.keras`, and `requirements.txt`) are located in the same folder.

**2. Install the required dependencies:**
Open your terminal (or command prompt), navigate to the project folder, and run the following command to install the required machine learning and computer vision libraries:
```bash
pip install -r requirements.txt
```

## Running

**Once the dependencies are successfully installed, execute the main script:**
```bash
python live_identity.py
```

## Interaction Guide:

**1. Silent Mode: When the camera detects no human subjects, the screen remains silent without any UI/bounding boxes.**

**2. Algorithmic Judgment: Step into the frame. The system will forcefully bisect your torso (Top 15%-50%, Bottom 50%-100%) and assign independent class labels in real-time.**

**3. Press the q key to quit the installation.**

## Note: The first time you run the script, it may take a few seconds to automatically download the YOLOv8 pre-trained weights (yolov8n.pt).
