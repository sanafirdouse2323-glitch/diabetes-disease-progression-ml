# Diabetes Disease Progression with Machine Learning

I built this project to understand how different machine-learning methods perform when estimating a measure of diabetes disease progression. I wanted to compare traditional regression models with tree-based methods and neural networks, and see whether using a more complex model actually leads to better results.

The project uses a continuous disease-progression target. It does not predict whether a person has diabetes, diagnose a condition, or recommend treatment.

## Dataset

I used the diabetes dataset provided by scikit-learn. It contains 442 observations and 10 preprocessed baseline features. The target is a continuous measure of disease progression one year after the baseline measurements.

This is a small public benchmark dataset, not a collection of raw hospital records. The demo included in this repository uses illustrative standardised inputs. They are synthetic examples and should not be treated as measurements from real patients.

## What I worked on

I prepared the data and compared several regression approaches: Linear Regression, Ridge, Lasso, ElasticNet, Support Vector Regression, Random Forest, Extra Trees and Histogram Gradient Boosting. I also implemented small neural networks using PyTorch and TensorFlow/Keras.

I evaluated the models using held-out test data and repeated train/test splits. I added nested cross-validation to examine whether selecting between model families improved performance. For prediction uncertainty, I implemented split-conformal prediction intervals, keeping the calibration data separate from the data used to fit the models. I also looked at prediction errors across the two encoded values of the dataset's standardised `sex` feature and examined model coefficients and feature importance.

I tried to keep the different stages of the experiments separate so that the test data was not used to fit the models or make tuning decisions. The detailed split sizes and experiment settings are recorded in [RESULTS.md](RESULTS.md).

## Results

The main finding was that the more complicated models did not automatically perform better on this dataset.

In the single held-out test run, Lasso had the lowest mean absolute error (MAE), at 42.703. In the ten-seed matched-sample comparison, Linear Regression had a mean test MAE of 42.899 and Lasso had 42.927. The default PyTorch and TensorFlow networks had higher mean test MAEs of 53.964 and 57.204.

I also ran a separate validation-based tuning experiment for the PyTorch network. Its mean test MAE was 44.934. This was an improvement over the default PyTorch network, but it did not show that the tuned network performed better than the simpler regression models. In the nested cross-validation experiment, the model-family selection procedure had a mean MAE of 44.854, compared with 44.483 for the fixed Linear Regression strategy. Model selection did not improve the result in that experiment.

These results come from different experiment designs and should not be combined into a single model ranking. The repeated test splits also overlap, so the standard deviations reported in the results are descriptive measures of variation between splits, not confidence intervals.

The conformal intervals and subgroup analysis also need to be interpreted carefully. Conformal coverage is a marginal guarantee under the relevant exchangeability assumptions; it is not a probability statement about an individual person. The subgroup analysis is exploratory and limited to the two encoded levels of one standardised feature. It does not establish that a model is fair. Similarly, feature coefficients and importance scores describe how a fitted model behaves; they do not show that a feature causes disease progression.

The full metrics, experiment settings and run records are available in [RESULTS.md](RESULTS.md).

## Running the project

The basic experiment uses the scikit-learn models and the core project dependencies. Python 3.10 or later is required.

Create a virtual environment and install the project:

```bash
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# Windows Command Prompt
.venv\\Scripts\\activate.bat

# Linux or macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Run the main experiment:

```bash
python -m diabetes_ml.experiment --output results
```

To run the neural-network experiment, install the optional neural dependencies:

```bash
pip install -e ".[neural]"
python -m diabetes_ml.neural.neural_experiment --output results --epochs 200 --patience 20
```

Other experiment commands are available for the repeated classical comparison, nested model comparison and repeated neural-network studies:

```bash
python -m diabetes_ml.repeated_validation --output repeated-validation
python -m diabetes_ml.nested_tabular_benchmark --output nested-tabular-benchmark
python -m diabetes_ml.repeated_neural_validation --output repeated-neural-validation
python -m diabetes_ml.repeated_neural_tuning --output repeated-neural-tuning
```

The neural commands require the optional neural dependencies. The interactive demo has its own optional dependencies:

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

To execute the notebooks, install the notebook and neural extras:

```bash
pip install -e ".[notebooks,neural]"
jupyter nbconvert --to notebook --execute notebooks/01_exploration.ipynb
```

To run the tests:

```bash
pip install -e ".[dev]"
pytest -v
```

## Project structure

The main Python package is in `src/diabetes_ml`. It contains the data preparation, models, evaluation methods and experiment scripts. The `tests` directory contains the automated tests. The `notebooks` directory contains the exploration, model comparison and neural-network notebooks. The Streamlit demonstration is in `demo/app.py`.

GitHub Actions workflows run the tests and experiment checks. They also build the Docker image, start the demo in a container and check that the application responds. The repository includes the Dockerfile and workflow definitions so these checks can be reviewed.

## Limitations and next steps

This project is an independent machine-learning study using a small, preprocessed public dataset. It has not been validated on an independent clinical dataset and is not intended for use in patient care. The results do not establish clinical usefulness or generalisation to other populations.

A meaningful next step would be to define a specific clinical research question and evaluate it using a larger, appropriately governed medical dataset. That work would need suitable patient-level data splitting, independent validation and a more complete assessment of model performance and fairness.

## References and licence

The dataset is available through scikit-learn's `load_diabetes` function and is associated with Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics.

This project is licensed under the MIT License.
