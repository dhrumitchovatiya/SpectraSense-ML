# SpectraSense-ML: Functional Group Identification using NMR + IR Spectroscopy and Machine Learning

## Project Overview

SpectraSense-ML is a machine learning project that predicts organic functional groups using multimodal spectroscopy data.

This project combines **chemistry domain knowledge** and **machine learning** to classify compounds based on their spectral signatures from:

* **¹H NMR (Nuclear Magnetic Resonance)**
* **IR (Infrared Spectroscopy)**

The goal is to automate functional group identification, a common task in organic chemistry and analytical chemistry.

---

## Problem Statement

Identifying functional groups manually from spectroscopy data can be difficult because many compounds have overlapping spectral regions.

Examples of overlapping IR peaks:

* **Ketone (C=O):** ~1715 cm⁻¹
* **Aldehyde (C=O):** ~1720–1740 cm⁻¹
* **Ester (C=O):** ~1735–1750 cm⁻¹

Because of this overlap, relying on a single spectroscopy feature often causes misclassification.

This project solves the problem by combining multiple spectral features and using machine learning for pattern recognition.

---

## Functional Groups Predicted

The model predicts 10 functional groups:

* Alcohol
* Aldehyde
* Ketone
* Ester
* Ether
* Amine
* Aromatic
* Alkene
* Alkyne
* Carboxylic Acid

---

## Features Used

### NMR Features

* Chemical Shift (ppm)
* Multiplicity
* Integration

### IR Features

* Main IR Peak (cm⁻¹)
* Secondary IR Peak (cm⁻¹)
* Peak Intensity

These features were selected because they provide complementary information about molecular structure.

---

## Dataset Design

The dataset used in this project is **synthetically generated but chemically realistic**.

It was intentionally designed with challenging overlap between similar functional groups to simulate real-world ambiguity.

Examples:

* Ketone vs Aldehyde vs Ester
* Alcohol vs Carboxylic Acid

This makes classification more realistic and prevents trivial prediction.

---

## Experimental Noise Simulation

Real spectroscopy instruments do not produce perfect values.

To simulate experimental uncertainty, controlled noise was added:

* **NMR variation:** ±0.2 ppm
* **IR variation:** ±10 cm⁻¹

This helps improve model robustness and avoids overfitting to idealized data.

---

## Machine Learning Model

Model used:

**Random Forest Classifier**

### Hyperparameters

* n_estimators = 250
* max_depth = 10
* min_samples_split = 5
* min_samples_leaf = 3
* random_state = 42

Random Forest was chosen because it performs well on nonlinear classification problems and provides feature importance analysis.

---

## Results

The model achieved approximately **95%+ accuracy** on the test dataset.

### Key Insights

* Most predictions were correct
* Small confusion observed in overlapping functional groups
* Feature importance aligned with chemistry knowledge

---

## Feature Importance

Model importance ranking:

1. IR Peak
2. NMR Shift
3. IR Secondary Peak
4. Peak Intensity
5. Integration
6. Multiplicity

This indicates IR and NMR peak positions are the strongest predictors.

---

## Visualizations

Project includes:

* Confusion Matrix
* Feature Importance Plot
* Functional Group Distribution
* NMR Shift Distribution
* IR Peak Distribution

These visualizations help interpret model performance and dataset characteristics.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Why This Project Matters

This project demonstrates:

* Machine Learning workflow
* Feature engineering
* Data preprocessing
* Model evaluation
* Domain-specific problem solving
* Application of AI in Chemistry

It highlights how machine learning can assist chemists in faster and more reliable spectroscopy interpretation.

---

## Future Improvements

Possible future enhancements:

* Use real spectroscopy datasets
* Add Mass Spectrometry features
* Deep learning approach
* Deploy as web application
* Real-time compound prediction tool

