"""Educational ML research demo for diabetes disease-progression regression.

Features:
- Point predictions with split-conformal marginal prediction intervals.
- Local feature attribution relative to the model-fitting subset.
- Exploratory subgroup error audit on the held-out benchmark test partition.

This application is not a diagnostic or clinical decision-support system.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation import (
    ConformalRegressor,
    compute_patient_explanation,
    evaluate_subgroup_disparity,
)
from diabetes_ml.models import get_classical_models


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
        page_title="Diabetes Progression Research Demo",
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

    # Sidebar: Model and Synthetic Feature Controls
    st.sidebar.header("⚙️ Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        ["Ridge", "Linear Regression", "Lasso", "Random Forest"],
        index=2,
        help="Select a regression model for the benchmark demonstration; Lasso had the strongest point metrics in the recorded split.",
    )

    st.sidebar.subheader("Synthetic Feature Profile")
    st.sidebar.caption("Illustrative standardized benchmark inputs only; these are not real patient measurements.")

    # Synthetic benchmark feature presets
    preset = st.sidebar.radio(
        "Load Preset Profile",
        ["Benchmark Mean", "Illustrative Profile A", "Illustrative Profile B"],
        index=0,
    )

    if preset == "Illustrative Profile A":
        default_vals = [0.05, 0.05, 0.08, 0.06, 0.04, 0.02, -0.04, 0.05, 0.08, 0.06]
    elif preset == "Illustrative Profile B":
        default_vals = [-0.04, -0.04, -0.06, -0.05, -0.03, -0.02, 0.06, -0.04, -0.07, -0.05]
    else:
        default_vals = [0.0] * 10

    feature_inputs = {}
    cols = st.sidebar.columns(2)
    for i, feat in enumerate(feature_names):
        col = cols[i % 2]
        feature_inputs[feat] = col.slider(
            feat,
            min_value=-0.15,
            max_value=0.15,
            value=float(default_vals[i]),
            step=0.01,
        )

    feature_vector = np.array([feature_inputs[f] for f in feature_names], dtype=np.float64)

    # Main Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Prediction & Conformal Interval",
        "🔍 Local Feature Attribution",
        "⚖️ Subgroup Error Audit",
    ])

    with tab1:
        st.subheader(f"Predictive Assessment: {selected_model_name}")

        cr = conformal_models[selected_model_name]
        intervals = cr.predict_interval(feature_vector.reshape(1, -1))
        pred_val = intervals.point_prediction[0]
        lower_bound = intervals.lower_bound[0]
        upper_bound = intervals.upper_bound[0]
        margin = intervals.margin

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Benchmark Target", f"{pred_val:.1f}")
        c2.metric("90% Conformal Lower Bound", f"{lower_bound:.1f}")
        c3.metric("90% Conformal Upper Bound", f"{upper_bound:.1f}")

        st.info(
            f"**90% split-conformal prediction interval:** **{lower_bound:.1f}** to "
            f"**{upper_bound:.1f}** (margin: ±{margin:.1f}). Under exchangeability, "
            "the procedure targets at least 90% marginal coverage across future observations. "
            "This is not a 90% probability statement about an individual observation."
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
        st.subheader("🔍 Local Feature Attribution Breakdown")
        st.caption("How each input feature shifts this synthetic profile's prediction relative to the model-fitting reference baseline.")

        explanation_df = compute_patient_explanation(
            models[selected_model_name],
            feature_vector,
            feature_names,
            ref_baseline,
        )

        st.dataframe(
            explanation_df.rename(columns={
                "feature": "Feature",
                "patient_value": "Feature Value",
                "deviation": "Deviation from Model-Fitting Mean",
                "contribution_to_progression": "Contribution to Predicted Score",
            }),
            use_container_width=True,
        )

        st.bar_chart(
            explanation_df.set_index("feature")["contribution_to_progression"],
            use_container_width=True,
        )

    with tab3:
        st.subheader("⚖️ Held-out Benchmark Subgroup Error Audit")
        st.markdown(
            """
            This exploratory audit compares errors across the two encoded values of the benchmark's standardized `sex` feature.
            These feature levels are not literal demographic group labels, and this small-sample audit does not establish fairness.
            """
        )

        test_preds = models[selected_model_name].predict(splits.X_test.values)
        sex_fairness = evaluate_subgroup_disparity(
            splits.y_test,
            test_preds,
            splits.X_test["sex"],
            subgroup_name="Standardized sex-feature level",
        )

        fc1, fc2 = st.columns(2)
        fc1.metric("Exploratory MAE Ratio", f"{sex_fairness['mae_parity_ratio']:.3f}", "1.0 = perfect parity")
        fc2.metric("Max Subgroup MAE Disparity", f"{sex_fairness['max_mae_disparity']:.2f}")

        st.json(sex_fairness["subgroups"])


if __name__ == "__main__":
    main()
