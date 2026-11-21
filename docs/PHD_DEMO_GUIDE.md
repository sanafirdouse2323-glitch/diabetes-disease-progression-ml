# PhD Demonstration Guide

## What this project is

I built an independent machine-learning project using a public diabetes disease-progression dataset. I compare different regression methods and examine how their predictions should be evaluated.

The dataset has 442 observations and 10 unstandardized source features. Its target is a continuous measure of disease progression one year after baseline. This is **not a diabetes diagnosis project**. The demos use synthetic values in the source feature representation, not real patient measurements.

## A simple 8–10 minute walkthrough

### 1. Explain the question

"I wanted to compare simple regression models, tree-based models and neural networks, and see whether the more complex methods actually improve the predictions."

### 2. Show the data

Open the README. Explain that the dataset is public and small, and that the loader returns unstandardized source features. Point out that the target is a progression measure, not a diabetes/no-diabetes label.

### 3. Explain the models

Briefly name the main model groups:

- Linear and regularised regression.
- Support Vector Regression.
- Tree-based models.
- Small neural networks using PyTorch and TensorFlow/Keras.

You do not need to explain every model in detail unless asked.

### 4. Explain how you checked the results

Show the experiment design in the research brief and [the fold-isolated preprocessing note](FOLD_ISOLATED_PREPROCESSING.md). Explain that linear-model scaling is learned only from model-fitting rows and separately within each cross-validation training fold. Mention that calibration and test data have separate roles. The numerical tables in RESULTS.md are historical and must not be quoted as results of the corrected implementation until the new workflows complete.

### 5. Discuss what you found

The previous experiment found that simpler regression models performed strongly and that a limited validation-selected PyTorch configuration improved on its default control. Those numerical findings belong to the earlier globally standardized input implementation. Do not quote them as current findings. Explain that the corrected experiment is being rerun and that results from different experiment designs are not directly comparable.

### 6. Show the demo and code

Run `streamlit run demo/app.py` after installing the demo extra. Use only the synthetic example values. Then show the package, tests and GitHub Actions workflows.

## A short interview explanation

"I independently built a machine-learning project using a public diabetes disease-progression dataset. I compared regression models, tree-based methods and small neural networks. I also worked on data splitting, model evaluation, prediction intervals and feature analysis. The results showed me that more complex models do not always perform better. This is a research demonstration, not a clinical tool, and I would like to apply the same methods to larger medical datasets with independent validation."

## Questions you may be asked

**Why did you choose this dataset?**

It is a public research dataset that allows me to practise and demonstrate regression methods without needing private hospital records.

**Does the project diagnose diabetes?**

No. The target is a continuous disease-progression measure. It is not a diagnosis label.

**Why include neural networks?**

I wanted to test whether they improved the results rather than assume they would. On this small dataset, the simpler models remained strong.

**Can this be used in a hospital?**

No. It has not been clinically validated and is not designed for diagnosis or treatment.

## Important points to remember

- Do not call this a diabetes diagnosis model.
- Do not claim that neural networks performed best.
- Do not describe prediction intervals as a patient's probability of an outcome.
- Do not claim the subgroup check proves fairness.
- Do not describe this small benchmark as clinical validation.
- Do not describe the separate 30-day hospital readmission classification experiment as external validation of the one-year progression regression task.
