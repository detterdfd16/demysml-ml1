import os
import numpy as np
import mne
import pywt

PROCESSED_ROOT =  "/Users/ricky/Library/CloudStorage/OneDrive-ImperialCollegeLondon/I-Explore/processed_all"
OUT_DIR = os.path.join(PROCESSED_ROOT, "_features_wavelet")
os.makedirs(OUT_DIR, exist_ok=True)

WAVELET = "db4" # Daubechies wavelet
LEVEL = 4 # Signal will be decomposed into 4 scales

subjects = sorted([d for d in os.listdir(PROCESSED_ROOT) if d.startswith("S")])

X_all = []
y_all = []
subj_all = []

for sid in subjects:
    fif_path = os.path.join(PROCESSED_ROOT, sid, f"{sid}-allruns-epo.fif")
    if not os.path.exists(fif_path):
        print("Missing:", fif_path)
        continue

    epochs = mne.read_epochs(fif_path, preload=True, verbose="error")
    X = epochs.get_data()      
    y = epochs.events[:, 2].astype(int)

    feats = []
    for i in range(X.shape[0]):  # each epoch
        f = []
        for ch in range(X.shape[1]):
            coeffs = pywt.wavedec(X[i, ch, :], wavelet=WAVELET, level=LEVEL) # Wavelet transform: output[cA4, cD4, cD3, cD2, cD1]
            for c in coeffs:
                f.append(np.sum(c * c))  # feature extraction: computes energy of each wavelet band
        feats.append(f)

    feats = np.array(feats, dtype=float)
    X_all.append(feats)
    y_all.append(y)
    subj_all.append(np.full(len(y), sid))

    print(sid, "done. X:", feats.shape, "y:", y.shape)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
subj_all = np.concatenate(subj_all, axis=0)

np.save(os.path.join(OUT_DIR, "X.npy"), X_all)
np.save(os.path.join(OUT_DIR, "y.npy"), y_all)
np.save(os.path.join(OUT_DIR, "subject.npy"), subj_all)

print("Saved to:", OUT_DIR)
print("Final X:", X_all.shape, "Final y:", y_all.shape)

# to used in model later 

# X = np.load("X.npy")
# y = np.load("y.npy")
# subject = np.load("subject.npy")

# then train them 