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
from sklearn.model_selection import train_test_split


@st.cache_resource
def load_trained_models():
    """Cache models and data splits for rapid interactive evaluation."""
    splits = create_experiment_splits(random_state=42)
    models = get_classical_models(random_state=42)

    # Keep calibration observations disjoint from model fitting.
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        splits.X_train,
        splits.y_train,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    conformal_models = {}
    for name, model in models.items():
        cr = ConformalRegressor(model, alpha=0.10)
        cr.fit_and_calibrate(
            X_fit.values,
            y_fit.values,
            X_cal.values,
            y_cal.values,
        )
        conformal_models[name] = cr

    return splits, models, conformal_models, X_fit


def main():
    st.set_page_config(
        page_title="Diabetes Progression Decision Support",
        page_icon="🩺",
        layout="wide",
    )

    st.title("🩺 Diabetes Disease Progression: Educational ML Research Demo")
    st.markdown(
        """
        An educational machine learning demonstration of **Split Conformal Prediction**
        and **local feature attribution** on a public, preprocessed benchmark dataset.
        This is not a diagnostic tool or a validated clinical decision system.
        """
    )

    splits, models, conformal_models, X_fit = load_trained_models()
    feature_names = splits.feature_names
    ref_baseline = X_fit.mean().values

    # Sidebar: Model and Patient Controls
    st.sidebar.header("⚙️ Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        ["Ridge", "Linear Regression", "Lasso", "Random Forest"],
        index=0,
        help="Select a regression model for the benchmark demonstration.",
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
            f"**90% split-conformal prediction interval:** **{lower_bound:.1f}** to "
            f"**{upper_bound:.1f}** (margin: ±{margin:.1f}). Under exchangeability, "
            "the procedure targets at least 90% marginal coverage across future observations. "
            "This is not a 90% probability statement about this individual patient."
        )

        # Describe the prediction against the benchmark target distribution only;
        # these labels are not clinical risk categories or treatment thresholds.
        q25, q75 = np.quantile(splits.y_train.to_numpy(dtype=float), [0.25, 0.75])
        st.caption("Position within the benchmark target distribution; not a clinical risk category.")
        if pred_val < q25:
            st.info("Prediction is below the benchmark training-set 25th percentile.")
        elif pred_val > q75:
            st.warning("Prediction is above the benchmark training-set 75th percentile.")
        else:
            st.success("Prediction is within the benchmark training-set interquartile range.")

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
