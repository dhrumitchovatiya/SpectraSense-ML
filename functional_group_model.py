# =========================================================
# CHALLENGING MULTIMODAL SPECTROSCOPY ML PROJECT
# REALISTIC OVERLAP + EXPERIMENTAL VARIATION
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "challenging_multimodal_spectroscopy_dataset.csv"
)

print("\n========================")
print("FIRST 5 ROWS")
print("========================\n")

print(df.head())

# =========================================================
# DATASET INFO
# =========================================================

print("\n========================")
print("DATASET INFO")
print("========================\n")

print(df.info())

# =========================================================
# CHECK MISSING VALUES
# =========================================================

print("\n========================")
print("MISSING VALUES")
print("========================\n")

print(df.isnull().sum())

# =========================================================
# HANDLE MISSING SECONDARY PEAKS
# =========================================================

# Fill missing secondary peaks
# using column median

df["IR_Secondary_Peak"] = (
    df["IR_Secondary_Peak"].fillna(
        df["IR_Secondary_Peak"].median()
    )
)

print("\n========================")
print("MISSING VALUES HANDLED")
print("========================\n")

print(df.isnull().sum())

# =========================================================
# ADD EXPERIMENTAL VARIATION
# =========================================================

# NMR SHIFT VARIATION
# ±0.2 ppm

nmr_noise = np.random.uniform(
    -0.2,
    0.2,
    size=len(df)
)

df["Shift"] = (
    df["Shift"] + nmr_noise
)

# IR MAIN PEAK VARIATION
# ±10 cm⁻¹

ir_noise = np.random.uniform(
    -10,
    10,
    size=len(df)
)

df["IR_Peak"] = (
    df["IR_Peak"] + ir_noise
)

# SECONDARY PEAK VARIATION
# ±10 cm⁻¹

secondary_noise = np.random.uniform(
    -10,
    10,
    size=len(df)
)

df["IR_Secondary_Peak"] = (
    df["IR_Secondary_Peak"] +
    secondary_noise
)

print("\n========================")
print("NOISE ADDED SUCCESSFULLY")
print("========================\n")

print(df.head())

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

# Multiplicity Encoding

multiplicity_encoder = LabelEncoder()

df["Multiplicity"] = (
    multiplicity_encoder.fit_transform(
        df["Multiplicity"]
    )
)

# Peak Intensity Encoding

intensity_encoder = LabelEncoder()

df["Peak_Intensity"] = (
    intensity_encoder.fit_transform(
        df["Peak_Intensity"]
    )
)

# Functional Group Encoding

target_encoder = LabelEncoder()

df["Functional_Group"] = (
    target_encoder.fit_transform(
        df["Functional_Group"]
    )
)

print("\n========================")
print("ENCODING COMPLETE")
print("========================\n")

print(df.head())

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[[
    "Shift",
    "Multiplicity",
    "Integration",
    "IR_Peak",
    "IR_Secondary_Peak",
    "Peak_Intensity"
]]

y = df["Functional_Group"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========================")
print("TRAIN TEST SPLIT COMPLETE")
print("========================\n")

print("Training Rows:", len(X_train))
print("Testing Rows:", len(X_test))

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(
    X_train,
    y_train
)

print("\n========================")
print("MODEL TRAINED SUCCESSFULLY")
print("========================\n")

# =========================================================
# PREDICTION
# =========================================================

predictions = model.predict(
    X_test
)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========================")
print("MODEL ACCURACY")
print("========================\n")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n========================")
print("CONFUSION MATRIX")
print("========================\n")

print(cm)

# =========================================================
# CONFUSION MATRIX HEATMAP
# =========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n========================")
print("CLASSIFICATION REPORT")
print("========================\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = (
    model.feature_importances_
)

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = (
    importance_df.sort_values(
        by="Importance",
        ascending=False
    )
)

print("\n========================")
print("FEATURE IMPORTANCE")
print("========================\n")

print(importance_df)

# =========================================================
# FEATURE IMPORTANCE PLOT
# =========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    x="Feature",
    y="Importance",
    data=importance_df
)

plt.title(
    "Feature Importance"
)

plt.xticks(rotation=45)

plt.show()

# =========================================================
# SAVE RESULTS
# =========================================================

results = X_test.copy()

results["Actual"] = (
    y_test.values
)

results["Predicted"] = (
    predictions
)

# Decode labels

results["Actual"] = (
    target_encoder.inverse_transform(
        results["Actual"]
    )
)

results["Predicted"] = (
    target_encoder.inverse_transform(
        results["Predicted"]
    )
)

# Save CSV

results.to_csv(
    "challenging_prediction_results.csv",
    index=False
)

print("\n========================")
print("RESULT FILE SAVED")
print("========================\n")

print(
    "challenging_prediction_results.csv"
)

# =========================================================
# SAMPLE PREDICTIONS
# =========================================================

print("\n========================")
print("SAMPLE PREDICTIONS")
print("========================\n")

print(
    results.head(15)
)

# =========================================================
# FUNCTIONAL GROUP DISTRIBUTION
# =========================================================

plt.figure(figsize=(12,6))

sns.countplot(
    x=target_encoder.inverse_transform(y)
)

plt.title(
    "Functional Group Distribution"
)

plt.xticks(rotation=45)

plt.show()

# =========================================================
# NMR SHIFT DISTRIBUTION
# =========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df["Shift"],
    bins=30,
    kde=True
)

plt.title(
    "NMR Chemical Shift Distribution"
)

plt.xlabel(
    "Shift (ppm)"
)

plt.show()

# =========================================================
# IR PEAK DISTRIBUTION
# =========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df["IR_Peak"],
    bins=30,
    kde=True
)

plt.title(
    "IR Peak Distribution"
)

plt.xlabel(
    "IR Peak (cm⁻¹)"
)

plt.show()

# =========================================================
# PROJECT COMPLETED
# =========================================================

print("\n========================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("========================\n")