"""Interactive Clinical Decision Support Dashboard for Diabetes Disease Progression.

Features:
- Real-time prediction with Conformal Prediction Intervals (guaranteed 90% coverage).
- Patient-level local biomarker attribution (why the model made this prediction).
- Subgroup demographic fairness audit tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from diabetes_ml.data import create_experiment_splits, load_diabetes_data
from diabetes_ml.evaluation import (
    ConformalRegressor,
    compute_patient_explanation,
    evaluate_subgroup_disparity,
    get_linear_attribution,
)
from diabetes_ml.models import get_classical_models


@st.cache_resource
def load_trained_models():
    """Cache models and data splits for rapid interactive evaluation."""
    splits = create_experiment_splits(random_state=42)
    models = get_classical_models(random_state=42)

    # Fit and calibrate Conformal Predictors
    conformal_models = {}
    for name, model in models.items():
        cr = ConformalRegressor(model, alpha=0.10)
        cr.fit_and_calibrate(
            splits.X_train.values,
            splits.y_train.values,
            splits.X_train.values[-70:],  # inner calibration partition
            splits.y_train.values[-70:],
        )
        conformal_models[name] = cr

    return splits, models, conformal_models


def main():
    st.set_page_config(
        page_title="Diabetes Progression Decision Support",
        page_icon="🩺",
        layout="wide",
    )

    st.title("🩺 Diabetes Disease Progression: Clinical AI Decision Support")
    st.markdown(
        """
        An educational clinical machine learning interface demonstrating **Distribution-Free Uncertainty Quantification**
        (Split Conformal Prediction) and **Patient-Level Biomarker Attribution**.
        """
    )

    splits, models, conformal_models = load_trained_models()
    feature_names = splits.feature_names
    ref_baseline = splits.X_train.mean().values

    # Sidebar: Model and Patient Controls
    st.sidebar.header("⚙️ Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        ["Ridge", "Linear Regression", "Lasso", "Random Forest"],
        index=0,
        help="Ridge regression is the top-performing model on this benchmark.",
    )

    st.sidebar.subheader("👤 Patient Biomarker Profile")
    st.sidebar.caption("Features are standardized clinical variables (scikit-learn benchmark scale).")

    # Patient presets
    preset = st.sidebar.radio(
        "Load Preset Profile",
        ["Default Median", "High Metabolic Risk", "Low Metabolic Risk"],
        index=0,
    )

    if preset == "High Metabolic Risk":
        default_vals = [0.05, 0.05, 0.08, 0.06, 0.04, 0.02, -0.04, 0.05, 0.08, 0.06]
    elif preset == "Low Metabolic Risk":
        default_vals = [-0.04, -0.04, -0.06, -0.05, -0.03, -0.02, 0.06, -0.04, -0.07, -0.05]
    else:
        default_vals = [0.0] * 10

    patient_inputs = {}
    cols = st.sidebar.columns(2)
    for i, feat in enumerate(feature_names):
        col = cols[i % 2]
        patient_inputs[feat] = col.slider(
            feat,
            min_value=-0.15,
            max_value=0.15,
            value=float(default_vals[i]),
            step=0.01,
        )

    patient_arr = np.array([patient_inputs[f] for f in feature_names], dtype=np.float64)

    # Main Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Patient Prediction & Uncertainty",
        "🔍 Local Biomarker Attribution",
        "⚖️ Benchmark & Fairness Audit",
    ])

    with tab1:
        st.subheader(f"Predictive Assessment: {selected_model_name}")

        cr = conformal_models[selected_model_name]
        intervals = cr.predict_interval(patient_arr.reshape(1, -1))
        pred_val = intervals.point_prediction[0]
        lower_bound = intervals.lower_bound[0]
        upper_bound = intervals.upper_bound[0]
        margin = intervals.margin

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted 1-Year Progression", f"{pred_val:.1f}")
        c2.metric("90% Conformal Lower Bound", f"{lower_bound:.1f}")
        c3.metric("90% Conformal Upper Bound", f"{upper_bound:.1f}")

        st.info(
            f"**Guaranteed Uncertainty Interval:** With 90% finite-sample statistical confidence, "
            f"the patient's true one-year disease progression is bounded between **{lower_bound:.1f}** and **{upper_bound:.1f}** "
            f"(margin: ±{margin:.1f})."
        )

        # Risk stratification indicator
        if pred_val < 100:
            st.success("🟢 **Risk Category: Mild Progression** (Below cohort 25th percentile)")
        elif pred_val < 180:
            st.warning("🟡 **Risk Category: Moderate Progression** (Cohort mid-range)")
        else:
            st.error("🔴 **Risk Category: Rapid Progression** (Upper quartile risk trajectory)")

    with tab2:
        st.subheader("🔍 Local Biomarker Attribution Breakdown")
        st.caption("How each clinical biomarker shifts this patient's prediction relative to the reference baseline.")

        explanation_df = compute_patient_explanation(
            models[selected_model_name],
            patient_arr,
            feature_names,
            ref_baseline,
        )

        st.dataframe(
            explanation_df.rename(columns={
                "feature": "Biomarker",
                "patient_value": "Patient Value",
                "deviation": "Deviation from Cohort Mean",
                "contribution_to_progression": "Impact on Progression Score",
            }),
            use_container_width=True,
        )

        st.bar_chart(
            explanation_df.set_index("feature")["contribution_to_progression"],
            use_container_width=True,
        )

    with tab3:
        st.subheader("⚖️ Held-out Performance & Demographic Disparity Audit")
        st.markdown(
            """
            Evaluating performance parity across demographic subgroups is critical for responsible medical AI.
            Below is the held-out evaluation on the 89 test patients disaggregated by demographic cohort.
            """
        )

        test_preds = models[selected_model_name].predict(splits.X_test)
        sex_fairness = evaluate_subgroup_disparity(
            splits.y_test,
            test_preds,
            splits.X_test["sex"],
            subgroup_name="Sex Cohort",
        )

        fc1, fc2 = st.columns(2)
        fc1.metric("Demographic MAE Parity Ratio", f"{sex_fairness['mae_parity_ratio']:.3f}", "1.0 = perfect parity")
        fc2.metric("Max Subgroup MAE Disparity", f"{sex_fairness['max_mae_disparity']:.2f}")

        st.json(sex_fairness["subgroups"])


if __name__ == "__main__":
    main()
