# Diabetes disease progression: a machine-learning study

I built this repository to compare regression methods on the public diabetes disease-progression dataset included in scikit-learn. The work covers data preparation, classical regression, small neural networks, model evaluation, prediction intervals and exploratory analysis of model errors.

The task is regression. The target is a continuous measure of disease progression one year after baseline. It is not a diabetes diagnosis, a prediction of whether someone has diabetes, or a recommendation for treatment.

## The dataset

The scikit-learn dataset contains 442 observations and 10 baseline features. The target is a quantitative disease-progression measure recorded one year after the baseline measurements. It is not a glucose reading, an HbA1c value or a diagnosis.

In the current implementation I load the dataset with `load_diabetes(as_frame=True, scaled=False)`. This returns the dataset's unstandardized feature representation and avoids calculating feature means and standard deviations across all 442 observations before the data is split. The source features are still a prepared benchmark representation; they are not raw hospital records.

## What I implemented

I compared Linear Regression, Ridge, Lasso, ElasticNet, Support Vector Regression, Random Forest, Extra Trees and Histogram Gradient Boosting. I also trained small multilayer perceptrons with PyTorch and TensorFlow/Keras.

For the classical models, I put `StandardScaler` inside the estimator pipeline for the scale-sensitive linear methods. That means the scaler is fitted on the observations used to fit the model. During cross-validation, each training fold gets its own fitted scaler. The neural-network workflow follows the same boundary: its scaler is fitted only on the model-fitting subset, then applied unchanged to the validation, calibration and test subsets. Tree-based models and the mean baseline do not need feature scaling.

I also implemented split-conformal prediction intervals, a small subgroup error audit across the two coded values of the dataset's `sex` feature, and model-based feature summaries. These analyses are exploratory. The subgroup comparison is not a complete fairness assessment, and model coefficients or feature importance do not establish cause and effect.

## Interactive demos

The repository contains four ways to explore the regression benchmark. The [GitHub Pages dashboard](https://sanafirdouse2323-glitch.github.io/diabetes-disease-progression-ml) is published from `main`; updates from this branch will appear there after the pull request is merged and the site is refreshed. The [Colab notebook](https://colab.research.google.com/github/sanafirdouse2323-glitch/diabetes-disease-progression-ml/blob/research/fold-isolated-preprocessing-20261121/notebooks/02_model_comparison.ipynb) runs the classical comparison in the cloud. The repository also includes a [Gradio interface](demo/gradio_app.py) and a [Streamlit app](demo/app.py) that you can run locally.

The sliders in the demos use illustrative values in the dataset's unstandardized source feature representation. They are not real patient measurements. A displayed prediction is a benchmark model output, not a personal disease-risk estimate or a clinical result. The conformal interval is designed to provide marginal coverage across future observations under exchangeability assumptions; it is not a 90% probability statement about one individual. The subgroup view is exploratory and does not prove fairness.

To run the Streamlit app locally, install the demo dependencies and start it from the repository root:

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

## Results from the corrected experiments

The results below were generated after the preprocessing change described above. The repeated hold-out experiments use seeds 42–51. In the matched-sample design, each seed has 225 model-fitting observations, 57 reserved validation observations, 71 calibration observations and 89 test observations. The 57-row validation subset is reserved but is not used to fit or select the classical models, so the sample sizes match the neural baseline design.

### Classical models and fixed neural baselines

The table shows mean test performance and sample standard deviation across the ten repeated splits. Lower MAE and RMSE are better; a higher R² is better.

| Model | Test MAE (mean ± SD) | Test RMSE (mean ± SD) | Test R² (mean ± SD) |
|---|---:|---:|---:|
| Ridge | **42.822 ± 2.870** | 52.893 ± 3.636 | 0.493 ± 0.064 |
| Lasso | 42.844 ± 2.845 | **52.890 ± 3.652** | 0.493 ± 0.065 |
| Linear Regression | 42.899 ± 2.836 | 52.975 ± 3.632 | 0.491 ± 0.064 |
| Random Forest | 44.851 ± 3.289 | 55.532 ± 3.580 | 0.442 ± 0.051 |
| PyTorch MLP (fixed baseline) | 53.963 ± 3.984 | 69.300 ± 4.493 | 0.127 ± 0.124 |
| TensorFlow MLP (fixed baseline) | 57.204 ± 8.404 | 73.242 ± 10.420 | 0.015 ± 0.261 |
| Mean baseline | 64.492 ± 3.627 | 74.945 ± 3.536 | −0.014 ± 0.015 |

For the neural baseline, I used a maximum of 100 epochs, patience of 12, batch size of 32, learning rate of 0.001 and dropout of 0.10. Both frameworks reached the epoch limit in all ten seeds. Under these settings and with this small dataset, the fixed neural baselines had higher mean errors than the classical regressors. This is a result for these experiments, not a general conclusion about neural networks.

### Validation-selected PyTorch study

I ran a separate ten-seed tuning study with a predeclared ten-candidate grid. The configuration was selected using validation MSE only. The calibration subset was kept separate, and the outer test subset was used only for the final evaluation.

| Metric | Validation-selected model (mean ± SD) | Paired default control (mean ± SD) |
|---|---:|---:|
| Test MAE | **43.408 ± 4.174** | 44.086 ± 3.365 |
| Test RMSE | **54.312 ± 4.278** | 55.714 ± 4.038 |
| Test R² | **0.464 ± 0.079** | 0.435 ± 0.086 |

Target standardisation was selected in seven of the ten seeds. The mean paired change in MAE was −0.677 ± 2.523 target units. The tuned model improved on its paired control, but its mean MAE remained higher than Ridge, Lasso and Linear Regression in the matched-sample comparison. It was lower than the Random Forest mean in this particular comparison. These are descriptive results, not a statistical proof that one model is superior.

The paired default control in this tuning study is not the same as the fixed 100-epoch neural baseline in the previous table; it comes from the expanded candidate study.

### Nested cross-validation

I also compared seven model families plus a mean baseline using five outer folds repeated three times. Within each outer training partition, three inner folds were used to tune the candidate models. All scaling for scale-sensitive models was fitted inside the corresponding inner training fold. This produced 15 outer-fold evaluations for each model family.

| Model family | Outer-fold test MAE (mean ± SD) |
|---|---:|
| SVR-RBF | **44.414 ± 3.270** |
| Linear Regression | 44.483 ± 2.626 |
| ElasticNet | 44.548 ± 2.601 |
| Ridge | 44.565 ± 2.606 |
| Extra Trees | 45.584 ± 3.105 |
| Random Forest | 46.449 ± 2.798 |
| Histogram Gradient Boosting | 46.524 ± 3.186 |
| Mean baseline | 65.896 ± 3.771 |

The nested procedure also selected a model family separately in each outer fold, using inner-fold MAE. SVR-RBF was selected in 10 of the 15 folds. The selected-family procedure had a mean outer-fold MAE of 44.854 ± 3.105, compared with 44.483 ± 2.626 for the fixed Linear Regression strategy. Across the 15 paired folds, the selector's MAE was higher in seven folds, lower in seven and equal in one. In this experiment, adding a model-selection step did not show a clear improvement over keeping Linear Regression fixed.

This nested benchmark does not calculate conformal intervals. Its results are not directly interchangeable with the repeated hold-out results above.

### Prediction intervals

The repeated classical experiment uses a separate 71-observation calibration subset for each seed and targets 90% marginal coverage. The mean empirical coverages were 91.8% for Lasso, 91.7% for Ridge, 91.6% for Linear Regression and 91.2% for Random Forest. Individual splits varied, and some observed coverages were below 90%.

Split-conformal coverage is a marginal property under exchangeability assumptions. It does not mean that a particular person's outcome has a 90% probability of falling inside the displayed interval.

### How to read these results

The standard deviations above describe variation across repeated splits. They are not confidence intervals, because the test partitions overlap across seeds. The nested outer folds also overlap across repeats. The experiments use different split designs, so their numbers should not be combined into a single league table. None of these results establishes clinical validity, clinical utility or performance in a new hospital or patient population.

The detailed per-seed results, calibration metrics, experiment settings and historical comparison tables are in [RESULTS.md](RESULTS.md). The current experiments were run from commit `e134cc1eff415da6d4847f8c4918e644b41978d7`. The associated [classical validation](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/44891212219), [neural validation](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/44891212166), [neural tuning](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/44891212238) and [nested-CV run](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/44891212335) completed successfully.

## Running the project

I use `uv` and the committed `uv.lock` when I need a repeatable environment. The GitHub Actions workflows use uv 0.13.8 and Python 3.11.

```bash
uv python install 3.11
uv sync --locked --no-build-isolation-package diabetes-ml --python 3.11 --extra build --extra dev
uv run --locked pytest -v
uv run --locked python -m diabetes_ml.experiment --output results
```

To install all optional components, including the neural, notebook and external-data dependencies, run:

```bash
uv sync --locked --no-build-isolation-package diabetes-ml --python 3.11 --all-extras
```

You can also install the project with pip, using the compatible version ranges in `pyproject.toml`:

```bash
python -m venv .venv
# Activate .venv for your operating system, then:
python -m pip install --upgrade pip
pip install -e .
```

The main experiment and the additional experiment commands are:

```bash
python -m diabetes_ml.experiment --output results
python -m diabetes_ml.repeated_validation --output repeated-validation
python -m diabetes_ml.nested_tabular_benchmark --output nested-tabular-benchmark
python -m diabetes_ml.neural.neural_experiment --output neural-results --epochs 200 --patience 20
python -m diabetes_ml.repeated_neural_validation --output repeated-neural-validation
python -m diabetes_ml.repeated_neural_tuning --output repeated-neural-tuning
```

The neural commands require the `neural` extra. The Streamlit app requires the `demo` extra. To execute the notebooks, install the `notebooks` extra as well. To run the test suite with pip, install the `dev` extra and run `pytest -v`.

## Separate study: 30-day hospital readmission

The repository also includes a separate classification experiment using the UCI *Diabetes 130-US Hospitals for Years 1999–2008* dataset (UCI dataset 296; DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J)). It works at the hospital-encounter level and studies recorded readmission within 30 days. This is a different dataset and prediction task. It is not external validation of the one-year disease-progression regression benchmark. The study design, data limitations and results are described in [READMISSION_STUDY_BRIEF.md](docs/READMISSION_STUDY_BRIEF.md).

## Repository layout

The `src/diabetes_ml/` directory contains the data loader, classical models, evaluation methods, neural trainers, repeated validation, nested cross-validation and the separate readmission experiment. The `demo/` directory contains the Streamlit and Gradio interfaces. The three notebooks cover data exploration, classical model comparison and neural-network training. Tests are in `tests/`, experiment documentation is in `docs/`, and the workflows in `.github/workflows/` run the automated checks and research experiments.

The repository also includes a dated record of an earlier neural CI run in `reports/ci-run-2026-11-02/`. Those files are retained for reproducibility and should not be treated as the latest results.

## Limitations and next steps

This is a small public benchmark with 442 observations. I have not evaluated it on an independent clinical cohort, raw hospital records or longitudinal patient data. The two coded values of the `sex` feature are not a comprehensive demographic fairness analysis. The current feature-attribution methods are model summaries and local perturbations; they are not SHAP explanations or causal analyses.

A useful next step would be to define a clinically meaningful research question and evaluate it on a larger, appropriately governed dataset, with patient-level splitting and independent validation. Any use of hospital data would require suitable access, privacy safeguards and research governance.

## References and licence

For the disease-progression dataset, see Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*, [The Annals of Statistics, 32(2), 407–499](https://doi.org/10.1214/009053604000000067), and the scikit-learn [`load_diabetes` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).

The separate readmission study uses the UCI dataset cited in its [research brief](docs/READMISSION_STUDY_BRIEF.md). This repository is released under the [MIT License](LICENSE).
