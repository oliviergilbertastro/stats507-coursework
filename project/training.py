from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
import joblib

tces = load_dataset("nyu-planet-finders/kepler-raw-lightcurves","tces",split="full",streaming=True,)

rows = [row for row in tces if row["av_training_set"] in {"PC", "AFP", "NTP"}] # Remove all the unknown from the data

df = pd.DataFrame(rows)

# Some properties without even looking at the lightcurves
feature_columns = [
    "tce_period",
    "tce_duration",
    "tce_depth",
    "tce_model_snr",
    "tce_max_mult_ev",
    "tce_impact",
    "tce_num_transits",
    "tce_prad",
]

X = df[feature_columns]
y = df["av_training_set"]
groups = df["kepid"]

# Hold out whole stars, avoiding the same star in both sets.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, test_indices = next(splitter.split(X, y, groups=groups))

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
)

model.fit(X.iloc[train_indices], y.iloc[train_indices])
predictions = model.predict(X.iloc[test_indices])

print("Test set results:")
print(classification_report(
    y.iloc[test_indices],
    predictions,
    labels=["PC", "AFP", "NTP"],
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(
    y.iloc[test_indices],
    predictions,
    labels=["PC", "AFP", "NTP"],
))

joblib.dump(model, "kepler_tce_rf.joblib")
print("Saved model to kepler_tce_rf.joblib")