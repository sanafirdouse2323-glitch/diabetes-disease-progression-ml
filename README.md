# Diabetes Disease Progression with Machine Learning

I built this project to compare different machine-learning methods on a public diabetes disease-progression dataset. I wanted to find out how well simple regression models, tree-based models and neural networks estimate the progression measure, and whether more complicated models actually improve the results.

This project predicts a **continuous disease-progression measure**. It does not predict whether someone has diabetes, diagnose a patient, or recommend treatment.

## Dataset

I use the diabetes dataset included with scikit-learn:

- 442 observations.
- 10 preprocessed baseline features.
- A continuous measure of disease progression one year after baseline.

The dataset is public and useful for learning and testing regression methods. It is small, and it is not raw hospital data. The interactive demo uses synthetic standardised inputs, not real patient measurements.

## What I did

I used the dataset to:

1. Prepare the data and train different regression models.
2. Compare their predictions using held-out test data and repeated experiments.
3. Check model selection with nested cross-validation.
4. Build prediction intervals using split-conformal prediction.
5. Compare prediction errors across the two encoded values of the standardised `sex` feature.
6. Examine model coefficients and feature importance to understand model behaviour.

The models include Linear Regression, Ridge, Lasso, ElasticNet, Support Vector Regression, Random Forest, Extra Trees, Histogram Gradient Boosting and small neural networks.

## What the results show

The results do not show that a more complicated model is automatically better.

- In the single held-out test run, Lasso had the lowest MAE: **42.703**.
- In the ten-seed matched-sample comparison, Linear Regression and Lasso had mean MAE values of **42.899** and **42.927**. The default PyTorch and TensorFlow networks had mean MAE values of **53.964** and **57.204**.
- A separate validation-selected PyTorch experiment reached mean MAE **44.934**. It came closer to the linear models but did not show that the neural network was better.
- In the nested cross-validation experiment, the model-family selection procedure had mean MAE **44.854**, compared with **44.483** for the fixed Linear Regression strategy. Selection did not improve the result in this experiment.

These numbers come from different experiment designs, so they should not be treated as one direct model ranking. Repeated test splits also overlap; their standard deviations are descriptive, not confidence intervals.

The full tables, experiment details and run records are in [RESULTS.md](RESULTS.md).

## Tools used

- **Python** for the project.
- **scikit-learn** for regression models, data preparation and evaluation.
- **NumPy and pandas** for working with the data.
- **Matplotlib** for plots.
- **PyTorch and TensorFlow/Keras** for the neural-network experiments.
- **Streamlit** for the interactive demo.
- **pytest, Docker and GitHub Actions** for testing and running the project.

The neural-network and demo packages are optional. You do not need them to run the basic scikit-learn experiment.

## Run it

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# Linux or macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Run the basic experiment:

```bash
python -m diabetes_ml.experiment --output results
```

For the neural-network experiments:

```bash
pip install -e ".[neural]"
python -m diabetes_ml.neural.neural_experiment --output results --epochs 200 --patience 20
```

For the repeated classical-model comparison:

```bash
python -m diabetes_ml.repeated_validation --output repeated-validation
```

For the nested model comparison:

```bash
python -m diabetes_ml.nested_tabular_benchmark --output nested-tabular-benchmark
```

For repeated neural-network experiments, install both neural frameworks first:

```bash
pip install -e ".[neural]"
python -m diabetes_ml.repeated_neural_validation --output repeated-neural-validation
```

For the validation-selected PyTorch tuning experiment:

```bash
python -m diabetes_ml.repeated_neural_tuning --output repeated-neural-tuning
```

For the interactive demo:

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

Run the tests:

```bash
pip install -e ".[dev]"
pytest -v
```

More experiment commands and setup details are available in the repository. The [PhD demonstration guide](docs/PHD_DEMO_GUIDE.md) explains which parts to show.

## What this project does not show

This is a small research and learning project, not a medical product. It has not been tested on an independent clinical dataset and should not be used for diagnosis or treatment. Its subgroup check is limited and does not prove that a model is fair. Feature importance describes model behaviour; it does not show that a feature causes disease progression.

A useful next step would be to apply the same careful evaluation to a larger medical dataset with a clearly defined clinical question and independent validation.

## Project documents

- [Research brief](docs/RESEARCH_BRIEF.md)
- [Detailed results](RESULTS.md)
- [PhD demonstration guide](docs/PHD_DEMO_GUIDE.md)

## Dataset reference and licence

The dataset is provided by scikit-learn through `load_diabetes`. It is associated with Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics.

This project is licensed under the MIT License.
