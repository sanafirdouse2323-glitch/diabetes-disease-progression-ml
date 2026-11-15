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

### Reproducible environment (recommended)

The committed `uv.lock` resolves the project and every declared optional extra, including the CPU PyTorch and TensorFlow stacks. Use the pinned uv release and locked sync to recreate an environment without re-resolving package versions:

```bash
# Install uv 0.13.8 using the official installer for your platform:
# https://docs.astral.sh/uv/getting-started/installation/

uv python install 3.11
uv sync --locked --no-build-isolation-package diabetes-ml --python 3.11 --extra build --extra dev --extra demo
uv run --locked pytest -v
uv run --locked python -m diabetes_ml.experiment --output results
```

To install every optional component, including neural, notebook, and external-dataset tooling:

```bash
uv sync --locked --no-build-isolation-package diabetes-ml --python 3.11 --all-extras
```

The lockfile pins resolved package versions and artifact hashes across supported environments. The Python interpreter, operating system, and external UCI data service remain separate runtime inputs; the external-data experiment records a SHA-256 fingerprint of the retrieved dataset tables.

### Install with pip (version ranges)

Python 3.10 through 3.13 is supported by the current lockfile. The following commands install compatible versions from the ranges in `pyproject.toml`; use the uv instructions above when exact dependency locking is required.

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

## External dataset study: hospital readmission (different target)

The repository includes a separate experiment using the UCI *Diabetes 130-US Hospitals for Years 1999-2008* dataset (UCI dataset 296; DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J)). UCI describes 101,766 hospital encounters from 130 US hospitals and a 30-day early-readmission classification task. The dataset is licensed CC BY 4.0. It contains demographic information, including age, gender, and race.

This study is **not** external validation of the original one-year disease-progression regression model: the cohort, unit of observation, outcome, and task differ. It is a related external-cohort classification study that broadens the project beyond the 442-row scikit-learn regression benchmark.

Install the optional data connector and run the experiment:

```bash
uv sync --locked --no-build-isolation-package diabetes-ml --python 3.11 --extra build --extra external-data
uv run --locked python -m diabetes_ml.external_readmission --output external-readmission-results --seeds 42 43 44
```

The experiment uses `readmitted == "<30"` as the positive label; `">30"` and `"NO"` are negative for the within-30-day endpoint. It uses `patient_nbr` only to form stratified patient-grouped folds and never as a model feature. All encounters for a patient stay in one partition. Race, gender, and age are excluded from the predictive feature matrix in this pre-specified baseline, and are retained only for exploratory subgroup reporting. Numeric/categorical preprocessing is fitted on each training partition. The model comparison includes a prior-only baseline, logistic regression, and a random forest.

The intended prediction time for this experiment is **at discharge**, because the predictor set includes discharge disposition and encounter-level information that is not necessarily available at admission. The experiment therefore should not be described as an admission-time prediction model.

This experiment uses the UCI label `NO` as a negative value because the source defines it as no recorded readmission. The underlying Health Facts database did not capture encounters from out-of-network providers, so `NO` must not be interpreted as proof that no readmission occurred anywhere. Readmission capture may be incomplete outside participating systems.

Threshold handling is explicit because the positive class is uncommon. A 0.5 cutoff is retained as a reference, not treated as an appropriate clinical operating point. For each outer split and model, a threshold is selected by maximising F2 (beta=2) on three-fold, patient-grouped out-of-fold probabilities generated only within the outer training partition. The selected threshold is then evaluated once on the untouched outer test partition. This avoids choosing a threshold from test labels. F2 weights recall more heavily than precision, but it is an experimental operating-point criterion, not a clinically justified decision rule. Results report discrimination and probability metrics as well as sensitivity, specificity, precision, and F2 at both the 0.5 reference cutoff and the training-selected threshold.

### Results from the current three-seed run

The positive class prevalence in the held-out folds was approximately 11.16%. The following are means across seeds 42–44; the thresholds were selected inside the corresponding outer training partitions and evaluated on their untouched test folds.

| Model | ROC-AUC | Average precision | Mean selected threshold | Precision at selected threshold | Sensitivity at selected threshold | Specificity at selected threshold | F2 at selected threshold | Sensitivity at 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior baseline | 0.5000 | 0.1116 | 0.11159 | 0.1116 | 1.0000 | 0.0000 | 0.3858 | 0.0000 |
| Logistic Regression | 0.6639 | 0.2127 | 0.07670 | 0.1424 | 0.8043 | 0.3916 | 0.4168 | 0.0213 |
| Random Forest | 0.6752 | 0.2195 | 0.10015 | 0.1477 | 0.7999 | 0.4197 | 0.4246 | 0.0000 |

The 0.5 cutoff therefore missed almost all positive encounters. The training-selected F2 thresholds raised test sensitivity to about 80%, but precision remained about 14–15% and specificity about 39–42%. In practical terms, the recall increase comes with many false-positive flags. Random Forest's mean F2 was only modestly above the prior baseline, and its ROC-AUC/average precision indicate limited discrimination rather than a strong predictive system. These results do not establish clinical usefulness.

There are only three outer seeds, and their test partitions overlap. The reported across-seed standard deviations are descriptive split-sensitivity summaries, not confidence intervals; no inferential claim of model superiority is made. Full per-seed, subgroup, and provenance outputs are retained as the GitHub Actions artifact for the run.

Only aggregate metrics and metadata are written to the output directory; raw patient-level data and patient identifiers are not written to the repository or workflow artifacts. The metadata records a SHA-256 fingerprint of a canonical serialization of the retrieved UCI tables so that a changed upstream copy can be detected. The across-seed standard deviations are descriptive split-sensitivity summaries because test folds may overlap across seeds. Subgroup metrics use the same training-selected threshold as the overall model and remain exploratory; they do not establish fairness, clinical utility, or generalisability to other health systems.

Dataset source and licence: [UCI Machine Learning Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008). Please cite Clore, J., Cios, K., DeShazo, J., and Strack, B. (2014), *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J).

The scope, prediction-time assumption, grouped evaluation, threshold-selection method, and limitations are documented separately in [READMISSION_STUDY_BRIEF.md](docs/READMISSION_STUDY_BRIEF.md).

## Project structure

The main Python package is in `src/diabetes_ml`. It contains the data preparation, models, evaluation methods and experiment scripts. The `tests` directory contains the automated tests. The `notebooks` directory contains the exploration, model comparison and neural-network notebooks. The Streamlit demonstration is in `demo/app.py`.

GitHub Actions workflows run the tests and experiment checks. They also build the Docker image, start the demo in a container and check that the application responds. The repository includes the Dockerfile and workflow definitions so these checks can be reviewed.

## Limitations and next steps

This project is an independent machine-learning study using a small, preprocessed public dataset and a separate historical hospital readmission dataset. The readmission study is a different prediction task and must not be interpreted as validating the disease-progression model. Neither study has been prospectively or independently validated for clinical use. The results do not establish clinical usefulness or support patient-care decisions.

Further research should evaluate the models with contemporary, independently governed cohorts, pre-specified clinical endpoints, patient- and site-aware validation, calibration, decision-curve analysis, and a more complete assessment of subgroup performance. Access to any additional hospital data requires appropriate governance, permissions, and privacy review.

## References and licence

**Disease-progression benchmark**

- Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). *Least Angle Regression*. The Annals of Statistics, 32(2), 407–499. [https://doi.org/10.1214/009053604000000067](https://doi.org/10.1214/009053604000000067).
- scikit-learn developers. *`sklearn.datasets.load_diabetes` API documentation*. [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).

**External readmission dataset**

- Clore, J., Cios, K., DeShazo, J., and Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5230J](https://doi.org/10.24432/C5230J).
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., and Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International, 2014, 781670. [https://doi.org/10.1155/2014/781670](https://doi.org/10.1155/2014/781670).

**Split-conformal prediction**

- Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). *Distribution-Free Predictive Inference for Regression*. Journal of the American Statistical Association, 113(523), 1094–1111. [https://doi.org/10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116).

This project is licensed under the MIT License.
