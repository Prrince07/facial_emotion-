# Real-Time Facial Emotion Detection

A facial-expression classification dashboard built with Python, OpenCV, DeepFace, and Streamlit. Detects faces in an image, draws bounding boxes, and estimates the probability of 7 facial expressions per face.

> **Note on terminology:** this system classifies *visible facial expressions* from image data. It does **not** determine a person's actual internal emotional state — a smile in a photo doesn't guarantee happiness, and the reverse is also true. Treat outputs as estimates, not facts.

## Overview

The app accepts an uploaded image or a webcam snapshot, detects every face present, and for each face predicts probabilities across:

`Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise`

The highest-probability class is shown as the **dominant emotion**, alongside a confidence score and a full probability breakdown, in a dark dashboard-style UI.

## Features

- Face detection with bounding boxes drawn directly on the image
- Multi-face support — each face analyzed and labeled independently
- 7-class facial-expression probability breakdown per face
- Dominant emotion + confidence score cards
- Image upload and camera-snapshot input
- Cached model loading (fast after first run)
- Input validation and clear error messages (no face found, invalid image, etc.)

## Tech Stack

| Component | Tool |
|---|---|
| UI / Dashboard | Streamlit |
| Face detection | OpenCV Haar Cascade |
| Emotion classification | DeepFace (pretrained) |
| Backend for DeepFace | TensorFlow |
| Image handling | OpenCV, Pillow, NumPy |

## How It Works

```
Image
  │
  ▼
Face Detection        (OpenCV Haar Cascade locates face regions)
  │
  ▼
Face Extraction        (crop each detected face, with a small margin)
  │
  ▼
Preprocessing           (resize/validate image before analysis)
  │
  ▼
Feature Extraction      (handled internally by DeepFace's CNN)
  │
  ▼
Emotion Classification  (DeepFace outputs a probability per class)
  │
  ▼
Probability Scores      (7 emotion percentages, normalized)
  │
  ▼
Dominant Emotion         (highest-probability class selected)
  │
  ▼
Dashboard                (Streamlit renders boxes, bars, and cards)
```

**Face Detection vs Face Recognition vs Facial Expression Recognition vs Emotion Classification:**
- *Face Detection* — finds *where* a face is in an image (a bounding box). No identity or expression involved.
- *Face Recognition* — identifies *who* the face belongs to (matches against known identities). Not used in this project.
- *Facial Expression Recognition* — analyzes the visual configuration of facial features (eyes, mouth, brows) to classify an expression category.
- *Emotion Classification* — the label applied to a facial expression's predicted category (e.g. "happy"). It's a classification output, not a measurement of felt emotion.

## Project Structure

```
facial-emotion-detection/
├── app.py                    # Streamlit UI (no ML logic here)
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── emotion_detector.py   # DeepFace wrapper — emotion probabilities
│   ├── face_detector.py      # OpenCV Haar Cascade wrapper — face boxes
│   ├── preprocessing.py      # Image validation & resizing
│   └── utils.py              # Drawing boxes, colors, formatting
├── models/                   # For a future custom-trained model
├── assets/                   # Icons/images if needed
└── screenshots/              # App screenshots for this README
```

## Installation

```bash
# 1. Create a virtual environment (Python 3.9–3.11 recommended)
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> If installation fails on TensorFlow/DeepFace, it's most likely a Python version conflict — see the note at the bottom of `requirements.txt`.

## Running the Project

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Model

- **Face detection:** OpenCV's `haarcascade_frontalface_default.xml` — a classical (non-deep-learning) Haar Cascade classifier shipped with OpenCV. Fast on CPU; works best on front-facing, reasonably well-lit faces.
- **Emotion classification:** [DeepFace](https://github.com/serengil/deepface), an open-source facial analysis library. Its emotion model is a CNN trained on the **FER-2013** dataset (grayscale, 48×48 facial expression images across 7 classes). DeepFace handles the model download and inference internally.
- **Confidence** = the probability percentage assigned to the dominant (highest-scoring) class. It reflects the model's relative certainty among the 7 classes, not an external ground-truth accuracy figure.

## Limitations

- Haar Cascade face detection can miss faces that are angled, partially occluded, or in poor lighting.
- FER-2013-trained models are known in the literature to struggle most with **Disgust** and **Fear** (fewer training examples, visually similar to other classes).
- A single static image cannot capture context (tone of voice, situation) that real emotional understanding relies on.
- Not validated for clinical, psychological, or security use.

## Future Improvements

- Emotion history / session timeline
- Live emotion graph over multiple frames
- CSV export of results
- Continuous real-time video mode (separate OpenCV-window script)
- Custom-trained CNN model (see below)
- Deployment guide for Streamlit Community Cloud

## Custom Model (Optional Next Step)

Once this pretrained version is working, a natural next step is training your own CNN:

- **Suggested dataset:** [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) (Kaggle) — 35,887 grayscale 48×48 images across the same 7 classes.
- **Split:** dataset already ships with train/test folders; carve out ~10% of train as validation.
- **Preprocessing:** grayscale, resize to 48×48, normalize pixel values to [0,1].
- **Augmentation:** horizontal flip, small rotations, zoom — helps generalization given FER-2013's noisy labels.
- **Architecture:** a small CNN — a few Conv2D + MaxPooling blocks, followed by Dense layers and a 7-way softmax output.
- **Loss:** categorical cross-entropy. **Optimizer:** Adam.
- **Evaluation:** accuracy, precision, recall, F1-score per class, and a confusion matrix (FER-2013 models typically land in the 60–70% accuracy range — this is a known dataset ceiling, not a code bug).
- **Save/load:** `model.save("models/emotion_cnn.h5")`, then load in `emotion_detector.py` in place of the DeepFace call — the rest of the app doesn't need to change.

## Screenshots

*(Add screenshots to the `screenshots/` folder and reference them here, e.g.)*

```
![Dashboard](screenshots/dashboard.png)
```

## Author

Your Name — BCA AI/ML Student
