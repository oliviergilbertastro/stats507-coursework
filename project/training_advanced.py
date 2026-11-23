from collections import Counter

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from time import perf_counter

# Parameters for the training
N_BINS = 100
PHASE_WINDOW = 0.1

repo = "nyu-planet-finders/kepler-raw-lightcurves"

tces = load_dataset(repo, "tces", split="full", streaming=True)

tce_by_kepid = {}

for tce in tces:
    if tce["av_training_set"] in {"PC", "AFP", "NTP"}:
        tce_by_kepid.setdefault(tce["kepid"], tce)

bin_sums = {kepid: np.zeros(N_BINS) for kepid in tce_by_kepid}
bin_counts = {kepid: np.zeros(N_BINS) for kepid in tce_by_kepid}

lightcurves = load_dataset(repo, "lightcurves", split="full", streaming=True)

rows_read = 0

total_rows = len(lightcurves)
start_time = perf_counter()
for row in lightcurves:
    rows_read += 1
    kepid = row["kepid"]
    tce = tce_by_kepid.get(kepid)

    if tce is None:
        continue

    period = tce["tce_period"]
    epoch = tce["tce_time0bk"]

    if not np.isfinite(period) or period <= 0 or not np.isfinite(epoch):
        continue

    time = np.asarray(row["time"], dtype=float)
    flux = np.asarray(row["pdcsap_flux"], dtype=float)
    quality = np.asarray(row["sap_quality"], dtype=int)
    good = (quality == 0) & np.isfinite(time) & np.isfinite(flux)
    time = time[good]
    flux = flux[good]

    if len(flux) == 0:
        continue

    median_flux = np.median(flux)

    if not np.isfinite(median_flux) or median_flux == 0:
        continue

    relative_flux = (flux / median_flux) - 1
    phase = ((time - epoch) / period + 0.5) % 1 - 0.5
    near_transit = np.abs(phase) <= PHASE_WINDOW
    phase = phase[near_transit]
    relative_flux = relative_flux[near_transit]

    if len(phase) == 0:
        continue

    bin_number = np.floor(
        (phase + PHASE_WINDOW) / (2 * PHASE_WINDOW) * N_BINS
    ).astype(int)
    bin_number = np.clip(bin_number, 0, N_BINS - 1)
    bin_sums[kepid] += np.bincount(
        bin_number,
        weights=relative_flux,
        minlength=N_BINS,
    )
    bin_counts[kepid] += np.bincount(bin_number, minlength=N_BINS)

    if rows_read % 5000 == 0:
        elapsed = perf_counter() - start_time
        percent = rows_read / total_rows * 100
        eta_seconds = elapsed / rows_read * (total_rows - rows_read)
        eta_minutes = int(eta_seconds / 60)
        print(f"{rows_read:,}/{total_rows:,} rows ({percent:.1f}%) | elapsed {elapsed / 60:.1f} min | estimated {eta_minutes} min remaining")

X = []
y = []
groups = []

for kepid, tce in tce_by_kepid.items():
    has_data = bin_counts[kepid] > 0

    if has_data.sum() < 10:
        continue

    binned_flux = np.full(N_BINS, np.nan)
    binned_flux[has_data] = bin_sums[kepid][has_data] / bin_counts[kepid][has_data]
    X.append(binned_flux)
    y.append(tce["av_training_set"])
    groups.append(kepid)

X = np.asarray(X)
y = np.asarray(y)
groups = np.asarray(groups)

print("Light-curve rows read:", rows_read)
print("Usable TCEs:", len(y))
print("Class counts:", Counter(y))

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
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