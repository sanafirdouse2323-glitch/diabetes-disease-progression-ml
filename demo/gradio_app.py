"""Interactive academic research demo for conformal prediction on a public regression benchmark.

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


def predict_progression(model_name: str, *feature_values: float):
    """Compute a benchmark prediction, conformal interval, and local attribution."""
    if len(feature_values) != len(feature_names):
        raise ValueError("The submitted feature count does not match the benchmark schema.")
    vector = np.asarray(feature_values, dtype=np.float64)
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
        f"*About the interval:* Under exchangeability between the calibration observations and future observations, "
        f"split conformal targets at least 90% marginal coverage. This is a statement about coverage across observations, "
        f"not a personal probability or a clinical risk estimate."
    )

    attr_df = explanation[["feature", "patient_value", "contribution_to_progression"]].copy()
    attr_df.columns = [
        "Feature",
        "Value (unstandardized source representation)",
        "Contribution to Score",
    ]
    attr_df["Contribution to Score"] = attr_df["Contribution to Score"].round(2)

    return summary_text, attr_df


# Gradio UI definition
with gr.Blocks(title="Diabetes ML: Conformal Uncertainty Estimation") as demo:
    gr.Markdown(
        """
        # Diabetes Progression: Exploring Predictions and Uncertainty
        ### A research demonstration using a continuous benchmark target
        *Based on [sanafirdouse2323-glitch/diabetes-disease-progression-ml](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml)*
        
        > **Academic note:** Inputs use the dataset's unstandardized source feature representation. The controls are synthetic examples, not patient measurements. This research demo does not provide diagnosis or treatment guidance and is not clinically validated.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=["Lasso", "Linear Regression", "Ridge", "Random Forest"],
                value="Lasso",
                label="Selected Model Family",
                info="Select a regression estimator for this educational benchmark demonstration.",
            )

            gr.Markdown("#### Unstandardized source feature inputs")
            feature_sliders = []
            for feature in feature_names:
                lower = float(X_fit[feature].min())
                upper = float(X_fit[feature].max())
                default = float(ref_baseline[list(feature_names).index(feature)])
                if feature == "sex":
                    default = float(round(default))
                    step = 1.0
                    label = "sex (coded source level)"
                else:
                    step = 0.001 if feature == "s5" else 0.1
                    default = float(np.clip(np.round(default / step) * step, lower, upper))
                    label = feature
                feature_sliders.append(
                    gr.Slider(
                        minimum=lower,
                        maximum=upper,
                        value=default,
                        step=step,
                        label=label,
                    )
                )

            submit_btn = gr.Button("Calculate Conformal Prediction", variant="primary")

        with gr.Column(scale=1):
            output_markdown = gr.Markdown(label="Conformal Prediction Results")
            attribution_table = gr.Dataframe(label="Local Feature Attribution Breakdown")

            gr.Markdown(
                """
                ---
                ### Research interpretation
                1. Linear, regularized, tree-based and neural methods are compared as benchmark estimators; no model family is assumed to be best.
                2. Split-conformal intervals target marginal coverage under exchangeability, not a personal probability statement.
                3. Learned preprocessing is fitted only within the model-fitting partition or the relevant cross-validation training fold.
                4. RESULTS.md separates the corrected fold-isolated experiments from the earlier pre-standardized results.
                """
            )

    submit_btn.click(
        fn=predict_progression,
        inputs=[model_selector, *feature_sliders],
        outputs=[output_markdown, attribution_table],
    )

if __name__ == "__main__":
    demo.launch()
