import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DentalVision AI™ | Clinical Diagnostics Suite",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .clinical-header {
        background: linear-gradient(135deg, #091e2f 0%, #0c2d48 50%, #145da0 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 15px 35px -10px rgba(12, 45, 72, 0.6);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-badge {
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    
    .clinical-title {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .clinical-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .metric-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 0.8rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    .result-container {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        border: 2px solid #0284c7;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px -15px rgba(2, 132, 199, 0.35);
        margin-bottom: 1.8rem;
    }
    
    .status-pill {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        padding: 5px 14px;
        border-radius: 50px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .diagnosis-headline {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
    }

    .confidence-meter {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 1rem;
    }

    .clinical-desc {
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.6;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 1rem;
    }

    .advice-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 4px solid #38bdf8;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .advice-card:hover {
        transform: translateX(4px);
        border-left-color: #0284c7;
    }
    .advice-text {
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

CLINICAL_DATABASE = {
    "Calculus": {
        "severity": "Moderate (Periodontal Prophylaxis Required)",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "#f59e0b",
        "badge_text": "#fbbf24",
        "icon": "🪥",
        "description": "Hardened, calcified bacterial plaque matrix adhering tenaciously to the coronal enamel surfaces and extending into subgingival margins.",
        "patient_advice": [
            "Schedule a professional ultrasonic scaling & root planing session to remove mineralized deposits.",
            "Switch to a tartar-control toothpaste fortified with pyrophosphates or zinc citrate to inhibit crystal formation.",
            "Maintain strict daily interdental cleaning using waxed dental floss or an interdental brush.",
            "Attend routine periodontal cleanings every 6 months to prevent localized attachment loss."
        ]
    },
    "Data caries": {
        "severity": "High Urgency (Restorative Intervention Mandatory)",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "#ef4444",
        "badge_text": "#f87171",
        "icon": "🦷",
        "description": "Progressive bacterial demineralization causing structural breakdown and cavitation within enamel and dentin tissues.",
        "patient_advice": [
            "Consult a restorative dentist immediately for cavity excavation and composite or ceramic restoration.",
            "Utilize high-concentration prescription fluoride toothpaste (1450+ ppm) to stimulate enamel remineralization.",
            "Strictly eliminate fermentable carbohydrate snacking and acidic carbonated beverages between meals.",
            "Rinse the oral cavity thoroughly with neutral water immediately following acidic food or drink exposure."
        ]
    },
    "Gingivitis": {
        "severity": "Reversible (Acute Soft-Tissue Inflammation)",
        "badge_bg": "rgba(249, 115, 22, 0.15)",
        "badge_border": "#f97316",
        "badge_text": "#fb923c",
        "icon": "🩸",
        "description": "Erythematous swelling, edema, and marginal bleeding of the gingival soft tissue elicited by undisturbed plaque biofilm.",
        "patient_advice": [
            "Adopt the Modified Bass tooth brushing technique at a 45° angle directed into the gingival sulcus.",
            "Perform short-term (10-14 days) antiseptic rinsing with 0.12% Chlorhexidine Gluconate or essential-oil formulations.",
            "Do not stop flossing if minor bleeding occurs; this is a natural consequence of resolving chronic hyperemia.",
            "Ensure regular vitamin C intake and stay well-hydrated to support mucosal tissue repair."
        ]
    },
    "Mouth Ulcer": {
        "severity": "Symptomatic (Self-Limiting / Mucosal Care)",
        "badge_bg": "rgba(56, 189, 248, 0.15)",
        "badge_border": "#38bdf8",
        "badge_text": "#38bdf8",
        "icon": "🩹",
        "description": "Painful focal loss of mucosal surface epithelium characterized by a necrotic fibrinous base surrounded by an erythematous halo.",
        "patient_advice": [
            "Apply topical hyaluronic acid barrier gel or triamcinolone acetonide paste to shield exposed nerve endings.",
            "Avoid thermal, chemical, or mechanical irritants including spicy, intensely citrus, and coarse food items.",
            "Rinse gently with isotonic warm saline (half a teaspoon of non-iodized salt in warm water) 3-4 times daily.",
            "If the ulcer persists beyond 14 days without spontaneous resolution, seek specialist consultation for diagnostic biopsy."
        ]
    },
    "Tooth Discoloration": {
        "severity": "Aesthetic Concern (Elective Correction)",
        "badge_bg": "rgba(168, 85, 247, 0.15)",
        "badge_border": "#a855f7",
        "badge_text": "#c084fc",
        "icon": "✨",
        "description": "Superficial extrinsic tannin/chromogen pigment absorption or intrinsic histological color deviation across the enamel layer.",
        "patient_advice": [
            "Undergo professional prophylaxis with prophy-jet air polishing to clear extrinsic stain deposits.",
            "Inquire about customized take-home carbamide peroxide trays or chairside in-office bleaching protocols.",
            "Limit direct prolonged enamel contact with chromogen-heavy beverages (black tea, espresso, dark wine, cola).",
            "Maintain post-intake water rinses to prevent chromogenic dyes from settling on enamel pores."
        ]
    },
    "hypodontia": {
        "severity": "Structural (Multidisciplinary Orthodontic / Prosthetic Plan)",
        "badge_bg": "rgba(99, 102, 241, 0.15)",
        "badge_border": "#6366f1",
        "badge_text": "#818cf8",
        "icon": "📐",
        "description": "Congenital developmental absence of one or more permanent tooth buds (excluding third molars).",
        "patient_advice": [
            "Undergo a comprehensive combined orthodontic and prosthetic clinical evaluation.",
            "Acquire panoramic radiography (OPG) or dental CBCT to inspect bone volume and neighboring root vectors.",
            "Discuss restorative pathways: orthodontic space closure, minimal-prep Maryland bridges, or endosseous dental implants.",
            "Preserve deciduous retaining teeth as long as functionally viable by minimizing interproximal decay."
        ]
    }
}

CLASS_NAMES = list(CLINICAL_DATABASE.keys())

@st.cache_resource
def load_classification_model():
    model_filename = "EfficientNetB0_best.keras"
    part_files = ["model_part1.bin", "model_part2.bin"]
    
    if not os.path.exists(model_filename):
        with open(model_filename, "wb") as output_stream:
            for part in part_files:
                if os.path.exists(part):
                    with open(part, "rb") as input_stream:
                        output_stream.write(input_stream.read())
                else:
                    raise FileNotFoundError(f"Missing model chunk: {part}")
                    
    return tf.keras.models.load_model(model_filename)

st.markdown("""
<div class="clinical-header">
    <div class="header-badge">🦷 Oral Pathology Diagnostic Suite</div>
    <div class="clinical-title">DentalVision AI™ Platform</div>
    <div class="clinical-subtitle">Automated Deep Learning Screening System Powered by Fine-Tuned EfficientNetB0</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏥 Clinical Specs & Audit")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">93.0%</div>
            <div class="metric-label">Macro Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">0.92</div>
            <div class="metric-label">Macro F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    **Core Network:** EfficientNetB0 (Fine-Tuned)  
    **Evaluation Baseline:** 2,464 Verified Clinical Annotations  
    **Loss Function:** AdamW Regularized Optimization  
    **Diagnostic Spectrum:** 6 Primary Pathology Classes  
    
    ---
    **⚠️ Clinical Governance Note:**  
    *This software operates as an assistive pre-screening tool designed to accelerate chairside triage. Official medical diagnoses must always be validated by a licensed Dental Surgeon.*
    """)
    st.caption("DentalVision Intelligence Systems © 2026")

col_input, col_output = st.columns([1.1, 1.4], gap="large")

with col_input:
    st.markdown("#### 📷 Intraoral Specimen Ingestion")
    uploaded_file = st.file_uploader(
        "Upload high-resolution intraoral capture (JPG, PNG, JPEG):",
        type=["jpg", "jpeg", "png"],
        help="Ensure consistent illumination and sharp focal depth on the lesion area."
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Current Diagnostic Specimen", use_container_width=True)
        st.caption("🔍 Image successfully normalized to 225×225 RGB tensor.")

with col_output:
    st.markdown("#### 🔬 Diagnostic Findings & Prescription")
    
    if uploaded_file is None:
        st.info("👈 Please upload an intraoral capture in the left panel to execute automatic feature extraction and classification.")
    else:
        with st.spinner("Executing deep convolutional feature maps & evaluating probability distribution..."):
            img_resized = image.resize((225, 225))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            model = load_classification_model()
            predictions = model.predict(img_array, verbose=0)[0]
            
            top_idx = int(np.argmax(predictions))
            top_class = CLASS_NAMES[top_idx]
            top_prob = float(predictions[top_idx]) * 100
            data = CLINICAL_DATABASE[top_class]

        st.markdown(f"""
        <div class="result-container">
            <span class="status-pill" style="background:{data['badge_bg']}; border:1px solid {data['badge_border']}; color:{data['badge_text']};">
                ● STATUS: {data['severity']}
            </span>
            <div class="diagnosis-headline">{data['icon']} {top_class}</div>
            <div class="confidence-meter">
                ⚡ Confidence Level: {top_prob:.1f}%
            </div>
            <div class="clinical-desc">
                <strong>Etiology & Morphological Characteristics:</strong><br>
                {data['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📋 Recommended Clinical Management & Patient Guidance")
        for advice in data["patient_advice"]:
            st.markdown(f"""
            <div class="advice-card">
                <p class="advice-text">🦷 <strong>Actionable Step:</strong> {advice}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("##### 📊 Differential Class Probabilities")
        sorted_indices = np.argsort(predictions)[::-1]
        for idx in sorted_indices:
            name = CLASS_NAMES[idx]
            score = float(predictions[idx])
            bar_col1, bar_col2 = st.columns([3, 1])
            with bar_col1:
                st.write(f"{CLINICAL_DATABASE[name]['icon']} **{name}**")
                st.progress(score)
            with bar_col2:
                st.write(f"**{score * 100:.1f}%**")
