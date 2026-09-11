"""
app.py
------
Streamlit UI for the Real-Time Facial Emotion Detection System.

This file is UI-only: it collects input from the user, calls into the
`src/` package to do the actual computer-vision work, and renders the
results as a dashboard. No ML/CV logic lives here.

Run with:
    streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st
from PIL import """
app.py
------
Streamlit UI for the Real-Time Facial Emotion Detection System.

This file is UI-only: it collects input from the user, calls into the
`src/` package to do the actual computer-vision work, and renders the
results as a dashboard. No ML/CV logic lives here.

Run with:
    streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.face_detector import detect_faces, crop_face
from src.emotion_detector import analyze_emotion, EMOTION_LABELS
from src.preprocessing import prepare_image
from src.utils import draw_face_box, get_emotion_color, get_emotion_box_color_bgr, format_probabilities_sorted


# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Facial Emotion Detector",
    page_icon="🎭",
    layout="wide",
)


# ----------------------------------------------------------------------
# STYLING (dark dashboard theme)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stat-card {
        background-color: #1A1F2B;
        border: 1px solid #2A2F3B;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .stat-label {
        color: #9AA4B2;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        color: #F5F7FA;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .emotion-row {
        margin-bottom: 14px;
    }
    .emotion-label {
        color: #E4E7EB;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .disclaimer {
        color: #7A8291;
        font-size: 0.8rem;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CACHED MODEL WARM-UP
# ----------------------------------------------------------------------
@st.cache_resource
def warm_up_model():
    """
    DeepFace loads its underlying TensorFlow model lazily on first use,
    which makes the *first* analysis slow. Running one tiny dummy
    analysis at app startup (cached so it only happens once per session)
    avoids that delay hitting the user's first real click.
    """
    try:
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        analyze_emotion(dummy_image)
    except Exception:
        # Warm-up failures are non-fatal - real analysis will raise
        # a clear error later if the model genuinely can't load.
        pass
    return True


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("🎭 AI Facial Emotion Detector")
st.markdown(
    "Upload an image or take a photo to detect faces and estimate facial "
    "expressions across 7 categories. This tool classifies **visible facial "
    "expressions** — it does not determine a person's true internal emotional state."
)

with st.spinner("Loading model..."):
    warm_up_model()

st.divider()


# ----------------------------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------------------------
input_col, results_col = st.columns([1, 1])

with input_col:
    st.subheader("Input")
    input_mode = st.radio("Choose input method", ["Upload Image", "Camera"], horizontal=True)

    uploaded_pil_image = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            uploaded_pil_image = Image.open(uploaded_file)
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            uploaded_pil_image = Image.open(camera_file)

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        st.rerun()

    image_placeholder = st.empty()
    if uploaded_pil_image is not None:
        image_placeholder.image(uploaded_pil_image, caption="Input preview", use_container_width=True)


# ----------------------------------------------------------------------
# ANALYSIS + RESULTS SECTION
# ----------------------------------------------------------------------
with results_col:
    st.subheader("Emotions")

    if not analyze_clicked:
        st.info("Upload or capture an image, then click **Analyze** to see results.")

    elif uploaded_pil_image is None:
        st.warning("Please provide an image before clicking Analyze.")

    else:
        try:
            with st.spinner("Analyzing image..."):
                image_bgr = prepare_image(uploaded_pil_image)
                faces = detect_faces(image_bgr)

                if len(faces) == 0:
                    st.error("No face detected. Try a clearer, front-facing, well-lit photo.")
                else:
                    annotated_image = image_bgr.copy()
                    face_results = []

                    for face in faces:
                        face_crop = crop_face(image_bgr, face)
                        result = analyze_emotion(face_crop)
                        face_results.append((face, result))

                        label = f"{result['dominant_emotion'].upper()} {result['confidence']:.0f}%"
                        box_color = get_emotion_box_color_bgr(result["dominant_emotion"])
                        annotated_image = draw_face_box(annotated_image, face, label, box_color)

                    # Show annotated image where the input preview was
                    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(annotated_rgb, caption="Detection result", use_container_width=True)

                    # --- Summary stat cards ---
                    primary_face, primary_result = face_results[0]
                    stat1, stat2, stat3 = st.columns(3)
                    with stat1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">Dominant Emotion</div>
                            <div class="stat-value">{primary_result['dominant_emotion'].capitalize()}</div>
                        </div>""", unsafe_allow_html=True)
                    with stat2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">Confidence</div>
                            <div class="stat-value">{primary_result['confidence']:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                    with stat3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">Faces Detected</div>
                            <div class="stat-value">{len(faces)}</div>
                        </div>""", unsafe_allow_html=True)

                    # --- Per-face probability bars ---
                    for idx, (face, result) in enumerate(face_results, start=1):
                        if len(face_results) > 1:
                            st.markdown(f"**Face {idx}**")

                        for emotion, value in format_probabilities_sorted(result["probabilities"]):
                            color = get_emotion_color(emotion)
                            st.markdown(f"""
                            <div class="emotion-row">
                                <div class="emotion-label">{emotion.upper()} — {value:.1f}%</div>
                                <div style="background-color:#2A2F3B; border-radius:6px; height:10px; width:100%;">
                                    <div style="background-color:{color}; width:{value}%; height:10px; border-radius:6px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("")

        except ValueError as validation_error:
            st.error(f"Invalid image: {validation_error}")
        except RuntimeError as model_error:
            st.error(f"Model error: {model_error}")
        except Exception as unexpected_error:
            st.error(f"Something went wrong: {unexpected_error}")


st.markdown(
    "<div class='disclaimer'>⚠️ Facial-expression predictions are estimates based on visual "
    "patterns and can be inaccurate. This tool is not a medical or psychological diagnostic "
    "instrument.</div>",
    unsafe_allow_html=True,
)


from src.face_detector import detect_faces, crop_face
from src.emotion_detector import analyze_emotion, EMOTION_LABELS
from src.preprocessing import prepare_image
from src.utils import draw_face_box, get_emotion_color, get_emotion_box_color_bgr, format_probabilities_sorted


# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Facial Emotion Detector",
    page_icon="🎭",
    layout="wide",
)


# ----------------------------------------------------------------------
# STYLING (dark dashboard theme)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stat-card {
        background-color: #1A1F2B;
        border: 1px solid #2A2F3B;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .stat-label {
        color: #9AA4B2;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        color: #F5F7FA;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .emotion-row {
        margin-bottom: 14px;
    }
    .emotion-label {
        color: #E4E7EB;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .disclaimer {
        color: #7A8291;
        font-size: 0.8rem;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CACHED MODEL WARM-UP
# ----------------------------------------------------------------------
@st.cache_resource
def warm_up_model():
    """
    DeepFace loads its underlying TensorFlow model lazily on first use,
    which makes the *first* analysis slow. Running one tiny dummy
    analysis at app startup (cached so it only happens once per session)
    avoids that delay hitting the user's first real click.
    """
    try:
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        analyze_emotion(dummy_image)
    except Exception:
        # Warm-up failures are non-fatal - real analysis will raise
        # a clear error later if the model genuinely can't load.
        pass
    return True


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("🎭 AI Facial Emotion Detector")
st.markdown(
    "Upload an image or take a photo to detect faces and estimate facial "
    "expressions across 7 categories. This tool classifies **visible facial "
    "expressions** — it does not determine a person's true internal emotional state."
)

with st.spinner("Loading model..."):
    warm_up_model()

st.divider()


# ----------------------------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------------------------
input_col, results_col = st.columns([1, 1])

with input_col:
    st.subheader("Input")
    input_mode = st.radio("Choose input method", ["Upload Image", "Camera"], horizontal=True)

    uploaded_pil_image = None

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            uploaded_pil_image = Image.open(uploaded_file)
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            uploaded_pil_image = Image.open(camera_file)

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        st.rerun()

    image_placeholder = st.empty()
    if uploaded_pil_image is not None:
        image_placeholder.image(uploaded_pil_image, caption="Input preview", use_container_width=True)


# ----------------------------------------------------------------------
# ANALYSIS + RESULTS SECTION
# ----------------------------------------------------------------------
with results_col:
    st.subheader("Emotions")

    if not analyze_clicked:
        st.info("Upload or capture an image, then click **Analyze** to see results.")

    elif uploaded_pil_image is None:
        st.warning("Please provide an image before clicking Analyze.")

    else:
        try:
            with st.spinner("Analyzing image..."):
                image_bgr = prepare_image(uploaded_pil_image)
                faces = detect_faces(image_bgr)

                if len(faces) == 0:
                    st.error("No face detected. Try a clearer, front-facing, well-lit photo.")
                else:
                    annotated_image = image_bgr.copy()
                    face_results = []

                    for face in faces:
                        face_crop = crop_face(image_bgr, face)
                        result = analyze_emotion(face_crop)
                        face_results.append((face, result))

                        label = f"{result['dominant_emotion'].upper()} {result['confidence']:.0f}%"
                        box_color = get_emotion_box_color_bgr(result["dominant_emotion"])
                        annotated_image = draw_face_box(annotated_image, face, label, box_color)

                    # Show annotated image where the input preview was
                    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(annotated_rgb, caption="Detection result", use_container_width=True)

                    # --- Summary stat cards ---
                    primary_face, primary_result = face_results[0]
                    stat1, stat2, stat3 = st.columns(3)
                    with stat1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">Dominant Emotion</div>
                            <div class="stat-value">{primary_result['dominant_emotion'].capitalize()}</div>
                        </div>""", unsafe_allow_html=True)
                    with stat2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">Confidence</div>
                            <div class="stat-value">{primary_result['confidence']:.1f}%</div>
                        </div>""", unsafe_allow_html=True)
                    with stat3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-label">Faces Detected</div>
                            <div class="stat-value">{len(faces)}</div>
                        </div>""", unsafe_allow_html=True)

                    # --- Per-face probability bars ---
                    for idx, (face, result) in enumerate(face_results, start=1):
                        if len(face_results) > 1:
                            st.markdown(f"**Face {idx}**")

                        for emotion, value in format_probabilities_sorted(result["probabilities"]):
                            color = get_emotion_color(emotion)
                            st.markdown(f"""
                            <div class="emotion-row">
                                <div class="emotion-label">{emotion.upper()} — {value:.1f}%</div>
                                <div style="background-color:#2A2F3B; border-radius:6px; height:10px; width:100%;">
                                    <div style="background-color:{color}; width:{value}%; height:10px; border-radius:6px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("")

        except ValueError as validation_error:
            st.error(f"Invalid image: {validation_error}")
        except RuntimeError as model_error:
            st.error(f"Model error: {model_error}")
        except Exception as unexpected_error:
            st.error(f"Something went wrong: {unexpected_error}")


st.markdown(
    "<div class='disclaimer'>⚠️ Facial-expression predictions are estimates based on visual "
    "patterns and can be inaccurate. This tool is not a medical or psychological diagnostic "
    "instrument.</div>",
    unsafe_allow_html=True,
)
