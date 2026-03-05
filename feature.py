import os
import numpy as np
import mne
import pywt
from sklearn.decomposition import PCA

PROCESSED_ROOT = "/Users/ricky/Library/CloudStorage/OneDrive-ImperialCollegeLondon/I-Explore/processed_all"

METHOD = "wavelet"
OUT_SUBFOLDER = f"_features_{METHOD}"

WAVELET = "db4" # Daubechies wavelet
LEVEL = 4 # decomposed into 4 scales: output[cA4, cD4, cD3, cD2, cD1]

N_COMPONENTS = 8 #output 8 principle components


def load_all_subjects(processed_root):
    subjects = sorted([d for d in os.listdir(processed_root) if d.startswith("S")])
    return subjects


def load_epochs_for_subject(processed_root, sid):
    fif_path = os.path.join(processed_root, sid, f"{sid}-allruns-epo.fif")
    if not os.path.exists(fif_path):
        return None
    epochs = mne.read_epochs(fif_path, preload=True, verbose="error")
    return epochs

# wavelet
def features_wavelet(X, wavelet="db4", level=4):
    feats = []
    for i in range(X.shape[0]):
        f = []
        for ch in range(X.shape[1]):
            coeffs = pywt.wavedec(X[i, ch, :], wavelet=wavelet, level=level) # Wavelet transform
            for c in coeffs:
                f.append(np.sum(c * c))  # Compute energy of each band
        feats.append(f)
    return np.array(feats, dtype=float)

# Riemannian
def features_cov_upper(X):
    feats = []
    for i in range(X.shape[0]):
        C = np.cov(X[i])  # (ch, ch)
        idx = np.triu_indices(C.shape[0])
        feats.append(C[idx])
    return np.array(feats, dtype=float)

# PCA
def features_logvar(X):

    var = np.var(X, axis=2) + 1e-12
    return np.log(var)


def main():
    out_dir = os.path.join(PROCESSED_ROOT, OUT_SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)

    subjects = load_all_subjects(PROCESSED_ROOT)
    print("Found subjects:", len(subjects))

    X_list, y_list, subj_list = [], [], []

    for sid in subjects:
        epochs = load_epochs_for_subject(PROCESSED_ROOT, sid)
        if epochs is None:
            print("Missing:", sid)
            continue

        X = epochs.get_data()                  # (n, ch, t)
        y = epochs.events[:, 2].astype(int)    # (n,)

        if METHOD == "wavelet":
            F = features_wavelet(X, WAVELET, LEVEL)

        elif METHOD == "cov":
            F = features_cov_upper(X)

        elif METHOD == "pca":
            F = features_logvar(X)

        else:
            raise ValueError("METHOD must be 'wavelet', 'cov', or 'pca'")

        X_list.append(F)
        y_list.append(y)
        subj_list.append(np.full(len(y), sid))

        print(sid, "done. F:", F.shape, "y:", y.shape)

    if len(X_list) == 0:
        raise RuntimeError("No data loaded.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    subj_all = np.concatenate(subj_list, axis=0)

    if METHOD == "pca":
        pca = PCA(n_components=N_COMPONENTS)
        X_all = pca.fit_transform(X_all)
        print("After PCA:", X_all.shape)

    np.save(os.path.join(out_dir, "X.npy"), X_all)
    np.save(os.path.join(out_dir, "y.npy"), y_all)
    np.save(os.path.join(out_dir, "subject.npy"), subj_all)

    print("Saved to:", out_dir)
    print("Final X:", X_all.shape, "Final y:", y_all.shape)


if __name__ == "__main__":
    main()

# # use in ML classifier

# X = np.load("X.npy")
# y = np.load("y.npy")
# subject = np.load("subject.npy")