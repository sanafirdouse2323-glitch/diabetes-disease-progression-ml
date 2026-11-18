"""Interactive Academic Research Demo for Conformal Prediction & Clinical ML Benchmarking.

Deployable to Hugging Face Spaces or runnable locally:
    pip install gradio scikit-learn pandas numpy
    python demo/gradio_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.models import get_classical_models
from diabetes_ml.evaluation import ConformalRegressor, compute_patient_explanation


# 1. Initialize benchmark models with split-conformal calibration
splits = create_experiment_splits(random_state=42)
models = get_classical_models(random_state=42)

# Keep calibration partition strictly disjoint from model fitting
X_fit, X_cal, y_fit, y_cal = train_test_split(
    splits.X_train,
    splits.y_train,
    test_size=0.20,
    random_state=42,
    shuffle=True,
)

conformal_predictors: dict[str, ConformalRegressor] = {}
for name, model in models.items():
    if name == "Mean Baseline":
        continue
    cr = ConformalRegressor(model, alpha=0.10)
    cr.fit_and_calibrate(
        X_fit.values,
        y_fit.values,
        X_cal.values,
        y_cal.values,
    )
    conformal_predictors[name] = cr

feature_names = splits.feature_names
ref_baseline = X_fit.mean().values


def predict_progression(
    model_name: str,
    age: float,
    sex: float,
    bmi: float,
    bp: float,
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    s5: float,
    s6: float,
):
    """Compute point prediction, 90% conformal prediction interval, and local feature attribution."""
    vector = np.array([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6], dtype=np.float64)
    cr = conformal_predictors[model_name]

    interval = cr.predict_interval(vector.reshape(1, -1))
    point_pred = float(interval.point_prediction[0])
    lower = float(interval.lower_bound[0])
    upper = float(interval.upper_bound[0])
    margin = float(interval.margin)

    # Explanation relative to baseline reference
    explanation = compute_patient_explanation(
        models[model_name],
        vector,
        feature_names,
        ref_baseline,
    )

    summary_text = (
        f"### Model: {model_name}\n\n"
        f"**Predicted 1-Year Progression Score:** `{point_pred:.1f}`\n\n"
        f"**90% Split-Conformal Prediction Interval:** `[{lower:.1f}, {upper:.1f}]` (Margin: ±{margin:.1f})\n\n"
        f"*Finite-Sample Coverage Property:* Under exchangeability between calibration and test data, "
        f"this interval is mathematically guaranteed to contain the true unobserved target with at least 90% marginal probability. "
        f"It is a marginal coverage guarantee, not a personal clinical probability statement."
    )

    attr_df = explanation[["feature", "patient_value", "contribution_to_progression"]].copy()
    attr_df.columns = ["Feature", "Standardized Value", "Contribution to Score"]
    attr_df["Contribution to Score"] = attr_df["Contribution to Score"].round(2)

    return summary_text, attr_df


# Gradio UI definition
with gr.Blocks(title="Diabetes ML: Conformal Uncertainty Estimation") as demo:
    gr.Markdown(
        """
        # 🩺 Clinical ML Benchmark: Conformal Uncertainty & Model Audit
        ### Research Walkthrough: Finite-Sample Prediction Intervals on Quantitative Disease Progression
        *Based on [sanafirdouse2323-glitch/diabetes-disease-progression-ml](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml)*
        
        > **Academic Note:** This tool evaluates regression methods on scikit-learn's standardized diabetes benchmark. 
        > Inputs are standardized benchmark variables, not raw medical records. It does not provide clinical diagnosis or treatment guidance.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=["Lasso", "Linear Regression", "Ridge", "Random Forest"],
                value="Lasso",
                label="Selected Model Family",
                info="Lasso and Linear Regression achieved the lowest mean MAE across repeated outer splits.",
            )

            gr.Markdown("#### Standardized Clinical Predictors")
            age_slider = gr.Slider(-0.15, 0.15, value=0.0, step=0.01, label="Age (standardized)")
            sex_slider = gr.Slider(-0.05, 0.05, value=0.0, step=0.01, label="Sex (standardized level)")
            bmi_slider = gr.Slider(-0.15, 0.15, value=0.04, step=0.01, label="Body Mass Index (BMI)")
            bp_slider = gr.Slider(-0.15, 0.15, value=0.02, step=0.01, label="Blood Pressure (BP)")
            s1_slider = gr.Slider(-0.15, 0.15, value=0.0, step=0.01, label="Serum Total Cholesterol (S1)")
            s2_slider = gr.Slider(-0.15, 0.15, value=0.0, step=0.01, label="Serum LDL (S2)")
            s3_slider = gr.Slider(-0.15, 0.15, value=-0.02, step=0.01, label="Serum HDL (S3)")
            s4_slider = gr.Slider(-0.15, 0.15, value=0.0, step=0.01, label="Serum Total/HDL Ratio (S4)")
            s5_slider = gr.Slider(-0.15, 0.15, value=0.05, step=0.01, label="Serum Triglycerides / Log(S5)")
            s6_slider = gr.Slider(-0.15, 0.15, value=0.0, step=0.01, label="Serum Glucose (S6)")

            submit_btn = gr.Button("Calculate Conformal Prediction", variant="primary")

        with gr.Column(scale=1):
            output_markdown = gr.Markdown(label="Conformal Prediction Results")
            attribution_table = gr.Dataframe(label="Local Feature Attribution Breakdown")

            gr.Markdown(
                """
                ---
                ### Key Research Findings from this Benchmark:
                1. **No Free Lunch:** Linear Regression (`MAE 42.899`) and Lasso (`MAE 42.927`) consistently outperformed small MLPs (`MAE 53.964`) on 225 training samples due to parameter-to-sample ratio bottlenecks ($P > N$).
                2. **Finite-Sample Uncertainty:** Split-conformal calibration delivers mathematically rigorous 90% coverage without assuming Gaussian errors.
                3. **Methodological Discipline:** All calibration sets and nested inner folds are isolated to prevent data leakage.
                """
            )

    submit_btn.click(
        fn=predict_progression,
        inputs=[
            model_selector, age_slider, sex_slider, bmi_slider, bp_slider,
            s1_slider, s2_slider, s3_slider, s4_slider, s5_slider, s6_slider
        ],
        outputs=[output_markdown, attribution_table],
    )

if __name__ == "__main__":
    demo.launch()
