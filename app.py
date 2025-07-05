# ------------------------------------------------------------
# 3-year Recurrence-Free Survival Predictor  (5-model RSF)
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ────────────────────────────────────────────────────────────
# 0. Load models  (cached so it loads only once)
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(path: str = "rsf_cv_models_np2.joblib"):
    artifact = joblib.load(path)
    return artifact["models"]

models = load_models()

# ────────────────────────────────────────────────────────────
# 1. Column order used at training
# ────────────────────────────────────────────────────────────
FEATURES = [
    "p_n_2", "p_t_2", "surgical_method2", "age", "cea_2", "ca19_9_2",
    "macro_cat", "bmi", "reconstruction2", "v_cat", "asa_ps_2", "sex",
    "diameter_o80", "hystological_cat", "location_tumor2", "pre_chemo2"
]

# ────────────────────────────────────────────────────────────
# 2. Encoding maps
# ────────────────────────────────────────────────────────────
asa_map        = {"1": 0, "2": 1, "3-4": 2}
surg_map       = {"DG": 0,"TG": 2, "PG": 1,}
recons_map     = {"B-1": 0, "B-2": 1, "R-Y": 2, "Other": 3}
macro_map      = {"Type 0": 0, "Type 1/2/3/5": 2, "Type 4": 3,"Unknown": 1}
v_map          = {"Negative": 0, "Positive": 2, "Unknown": 1}
p_t_map        = {"pT0": 0, "pT1": 1, "pT2": 2, "pT3": 3, "pT4": 4}
p_n_map        = {"pN0": 0, "pN1": 1, "pN2": 2, "pN3": 3}
pre_chemo_map  = {"no": 0, "yes": 1}

sex_map        = {"Male": 1, "Female": 0}
diameter_map   = {"<80mm": 0, "≥80mm": 2, "Unknown": 1}
histology_map  = {"pap/tub": 0, "por/sig/muc": 1}
location_map   = {"EG": 0, "U": 3, "M": 2, "L": 1}

# ────────────────────────────────────────────────────────────
# 3. Title (single line, smaller font)
# ────────────────────────────────────────────────────────────
st.markdown(
    "<h3 style='text-align:center;margin-bottom:1rem;'>3-year Recurrence-Free Survival Predictor</h3>",
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────
# 4. Input widgets (in requested order)
# ────────────────────────────────────────────────────────────
age_str = st.text_input("Age (years)")

sex = st.selectbox("Sex", sex_map.keys())

height_str = st.text_input("Height (cm)")
weight_str = st.text_input("Weight (kg)")

# BMI auto-calculation
bmi_val = None
if height_str and weight_str:
    try:
        h = float(height_str)
        w = float(weight_str)
        bmi_val = w / (h / 100) ** 2
        st.info(f"Calculated BMI: **{bmi_val:.1f}**")
    except ValueError:
        st.warning("Height and Weight must be numeric.")

cea_str   = st.text_input("CEA (ng/mL)")
ca199_str = st.text_input("CA19-9 (U/mL)")

prechemo = st.selectbox("Neoadjuvant chemo", pre_chemo_map.keys())
asa      = st.selectbox("ASA-PS", asa_map.keys())

# ── Tumor location ➔ limits on Surgical procedure ───────────
# ── Tumor location ➔ limits on Surgical procedure ───────────
location = st.selectbox("Tumor location", list(location_map.keys()))

_all_surg = ["DG", "TG", "PG"]

if location == "EG":          # EG → DG を除外
    surg_options = ["TG", "PG"]
elif location in ("L", "M"):  # L または M → PG を除外
    surg_options = ["DG", "TG"]
else:                         # U → 制限なし
    surg_options = _all_surg

surg = st.selectbox("Surgical procedure", surg_options)


# ── Surgical procedure ➔ limits on Reconstruction ──────────
if surg == "PG":
    recons_options = ["Other"]
elif surg == "TG":
    recons_options = ["R-Y", "Other"]
else:  # DG
    recons_options = ["B-1", "B-2", "R-Y", "Other"]
recons = st.selectbox("Reconstruction", recons_options)

macro     = st.selectbox("Macroscopic type", macro_map.keys())
diameter  = st.selectbox("Tumor diameter", diameter_map.keys())
histology = st.selectbox("Histology", histology_map.keys())

pt   = st.selectbox("Pathological T", p_t_map.keys())
pn   = st.selectbox("Pathological N", p_n_map.keys())
vcat = st.selectbox("Vascular invasion (v)", v_map.keys())

# ────────────────────────────────────────────────────────────
# 5. Prediction
# ────────────────────────────────────────────────────────────
if st.button("Predict"):
    # ―― numeric conversion & validation ――――――――――――――――
    try:
        age    = int(age_str)
        height = float(height_str)
        weight = float(weight_str)
        bmi    = weight / (height / 100) ** 2
    except ValueError:
        st.error("Age, Height, and Weight must be numeric.")
        st.stop()

    try:
        cea   = float(cea_str)
        ca199 = float(ca199_str)
    except ValueError:
        st.error("CEA and CA19-9 must be numeric.")
        st.stop()

    # ―― assemble DataFrame ―――――――――――――――――――――――――――
    inp = pd.DataFrame([{
        "p_n_2": p_n_map[pn],
        "p_t_2": p_t_map[pt],
        "surgical_method2": surg_map[surg],
        "age": age,
        "cea_2": cea,
        "ca19_9_2": ca199,
        "macro_cat": macro_map[macro],
        "bmi": bmi,
        "reconstruction2": recons_map[recons],
        "v_cat": v_map[vcat],
        "asa_ps_2": asa_map[asa],
        "sex": sex_map[sex],
        "diameter_o80": diameter_map[diameter],
        "hystological_cat": histology_map[histology],
        "location_tumor2": location_map[location],
        "pre_chemo2": pre_chemo_map[prechemo]
    }])[FEATURES]

    # ―― survival curve prediction ――――――――――――――――――――
    time_grid = np.arange(0, 37)            # 0-36 months
    surv_mat  = [
        np.interp(time_grid, fn.x, fn.y)
        for m in models for fn in m.predict_survival_function(inp)
    ]
    surv_mean = np.column_stack(surv_mat).mean(axis=1)

    rfs36 = float(surv_mean[time_grid == 36]) * 100
    st.success(f"Predicted 3-year RFS: **{rfs36:.1f}%**")

    # ―― plot curve ―――――――――――――――――――――――――――――――――――
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_grid, surv_mean, lw=2)
    ax.set_xlabel("Months after surgery")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 37, 6))
    ax.grid(alpha=0.3)
    st.pyplot(fig)
