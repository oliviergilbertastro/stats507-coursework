from collections import Counter
from itertools import islice

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

# Parameters for the training
MAX_LIGHTCURVE_ROWS = None #2000 # nb of lightcurves used for the training
N_BINS = 100
PHASE_WINDOW = 0.1  # Use data within 0.1 orbital cycles of transit


# Load labeled TCEs. For this first prototype, im keeping only one per star. UNK events are excluded.
tces = load_dataset("nyu-planet-finders/kepler-raw-lightcurves","tces",split="full",streaming=True,)

tce_by_kepid = {}

for tce in tces:
    if tce["av_training_set"] in {"PC", "AFP", "NTP"}:
        tce_by_kepid.setdefault(tce["kepid"], tce)

print("Labeled stars in TCE catalog:", len(tce_by_kepid))



lightcurves = load_dataset("nyu-planet-finders/kepler-raw-lightcurves","lightcurves",split="full",streaming=True,)

lightcurves_by_kepid = {}

for row in islice(lightcurves, MAX_LIGHTCURVE_ROWS):
    lightcurves_by_kepid.setdefault(row["kepid"], []).append(row)

print("Light-curve rows read:", MAX_LIGHTCURVE_ROWS if MAX_LIGHTCURVE_ROWS is not None else len(lightcurves))
print("Unique stars in those rows:", len(lightcurves_by_kepid))


# For each matching star, fold its flux measurements around the TCE
# period and epoch, then summarize the shape in 100 phase bins.
feature_rows = []
labels = []
groups = []

for kepid, quarter_rows in lightcurves_by_kepid.items():
    tce = tce_by_kepid.get(kepid)

    if tce is None:
        continue

    period = tce["tce_period"]
    epoch = tce["tce_time0bk"]

    if not np.isfinite(period) or period <= 0 or not np.isfinite(epoch):
        continue

    times = []
    fluxes = []
    qualities = []

    for row in quarter_rows:
        times.append(np.asarray(row["time"], dtype=float))
        fluxes.append(np.asarray(row["pdcsap_flux"], dtype=float))
        qualities.append(np.asarray(row["sap_quality"], dtype=int))

    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    quality = np.concatenate(qualities)

    # Keep measurements with no quality flags and finite time/flux.
    good = ((quality == 0) & np.isfinite(time) & np.isfinite(flux))
    time = time[good]
    flux = flux[good]

    if len(flux) == 0: # If somehow all the lightcurve is bad
        continue

    # For training, brightness relative to the star's typical brightness so the model focuses on the light-curve shape.
    median_flux = np.median(flux)

    if not np.isfinite(median_flux) or median_flux == 0:
        continue

    relative_flux = (flux / median_flux) - 1

    # Folding the time series so the predicted transit center is phase 0.
    phase = ((time - epoch) / period + 0.5) % 1 - 0.5

    # Keeping a window around the predicted transit.
    near_transit = np.abs(phase) <= PHASE_WINDOW
    phase = phase[near_transit]
    relative_flux = relative_flux[near_transit]

    if len(phase) == 0:
        continue

    # Putting measurements into some fixed phase bins. Empty bins should stay nan and are filled by the model's SimpleImputer.
    bin_number = np.floor(
        (phase + PHASE_WINDOW) / (2 * PHASE_WINDOW) * N_BINS
    ).astype(int)
    bin_number = np.clip(bin_number, 0, N_BINS - 1)

    binned_flux = np.full(N_BINS, np.nan)

    for bin_index in range(N_BINS):
        in_bin = bin_number == bin_index
        if np.any(in_bin):
            binned_flux[bin_index] = np.median(relative_flux[in_bin])

    # Skip curves with very little phase coverage.
    if np.isfinite(binned_flux).sum() < 10:
        continue

    feature_rows.append(binned_flux)
    labels.append(tce["av_training_set"])
    groups.append(kepid)


X = np.asarray(feature_rows)
y = np.asarray(labels)
groups = np.asarray(groups)

print("TCEs with usable light curves:", len(y))
print("Class counts:", Counter(y))

if len(y) < 10 or len(set(y)) < 2:
    raise ValueError(
        "Too few usable examples or classes in this limited sample. "
        "Try increasing MAX_LIGHTCURVE_ROWS."
    )


# Keep every star entirely in either the training or test set.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)
train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
)

model.fit(X[train_indices], y[train_indices])
predictions = model.predict(X[test_indices])

class_names = ["PC", "AFP", "NTP"]

print("\nTest set results:")
print(classification_report(
    y[test_indices],
    predictions,
    labels=class_names,
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(
    y[test_indices],
    predictions,
    labels=class_names,
))

joblib.dump(model, "kepler_lightcurve_rf.joblib")
print("\nSaved model to kepler_lightcurve_rf.joblib")