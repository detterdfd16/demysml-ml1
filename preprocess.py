import os
import re
import pathlib
import mne
import numpy as np

# Test trail times
# raw = mne.io.read_raw_edf(
#     r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\files\S001\S001R04.edf",
#     preload=True
# )
# events, event_id = mne.events_from_annotations(raw)

# sfreq = raw.info['sfreq']
# onsets = events[:, 0] / sfreq  # convert samples → seconds

# # compute time between consecutive events
# intervals = np.diff(onsets)

# print(intervals[:20])
# print("Mean interval:", intervals.mean())


lrRun = {3, 4, 7, 8, 11, 12}
bothRun = {5, 6, 9, 10, 13, 14}

def extract_run_number(edf_path: str) -> int:
    name = pathlib.Path(edf_path).name
    m = re.search(r"R(\d+)\.edf$", name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse run number from filename: {name}")
    return int(m.group(1))

def preprocess_one_edf(edf_path, out_fif_path, l_freq=8.0, h_freq=40.0,tmin=0.0, tmax=4.0,run_ica=True, random_state=1):

    edf_path = pathlib.Path(edf_path)
    out_fif_path = pathlib.Path(out_fif_path)
    out_fif_path.parent.mkdir(parents=True, exist_ok=True)

    run = extract_run_number(str(edf_path))

    raw = mne.io.read_raw_edf(edf_path, preload=True)

    raw.pick_types(eeg=True)
    raw.filter(l_freq, h_freq)
    raw.set_eeg_reference("average", projection=False, verbose="error")

    print(f"channel names: {raw.ch_names}")

    # ICA
    if run_ica:
        ica = mne.preprocessing.ICA(n_components=0.99,method="fastica",random_state=random_state, max_iter="auto",)
        ica.fit(raw, verbose="error")
        frontal = [ch for ch in ["Fp1", "Fp2", "AFp1", "AFp2"] if ch in raw.ch_names]
        if len(frontal) >= 1:
            eog_proxy = raw.copy().pick_channels(frontal).get_data().mean(axis=0)
            sources = ica.get_sources(raw).get_data()
            corr = np.array([np.corrcoef(sources[i], eog_proxy)[0, 1] for i in range(sources.shape[0])])
            bad_ics = np.where(np.abs(corr) > 0.3)[0].tolist() 
            ica.exclude = bad_ics
            raw = ica.apply(raw, verbose="error")
        else:
            pass

    # Events / epochs
    events, event_id = mne.events_from_annotations(raw)

    labels = {"T0", "T1", "T2"}
    if not labels.issubset(event_id.keys()):
        raise RuntimeError(f"[{edf_path.name}] Missing {labels - set(event_id.keys())}. Found: {list(event_id.keys())}")
    if run in lrRun:
        semantic = {"T0": "rest","T1": "left_fist","T2": "right_fist"}
    elif run in bothRun:
        semantic = {"T0": "rest","T1": "both_fists","T2": "both_feet"}
    else:
        raise ValueError(f"{edf_path}: run R{run:02d} not in task runs (3–14).")
    
    sem_event_id = {semantic[k]: event_id[k] for k in ["T0", "T1", "T2"]}

    epochs = mne.Epochs(
        raw, events, event_id=sem_event_id,
        tmin=tmin, tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="error"
    )
    epochs.save(out_fif_path, overwrite=True)
    return epochs


# Single run test
# edf_path = r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\files\S001\S001R04.edf"
# out_fif  = r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\ml1\demysml-ml1\processed\S001R04-epo.fif"
# epochs = preprocess_one_edf(edf_path, out_fif, run_ica=True)
# print(epochs)
# print("Epoch event_id:", epochs.event_id)

root_dir = pathlib.Path(r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\files")
processed_root = pathlib.Path(r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\ml1\demysml-ml1\processed_all")
processed_root.mkdir(parents=True, exist_ok=True)

by_subject = {}

for subject_dir in sorted(root_dir.glob("S*")):
    subject_id = subject_dir.name  # "S001"
    epochs_list = []

    for edf_file in sorted(subject_dir.glob(f"{subject_id}R*.edf")):
        run = extract_run_number(str(edf_file))
        if 3 <= run <= 14:
            out_dir = processed_root / subject_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_fif = out_dir / f"{edf_file.stem}-epo.fif"

            try:
                ep = preprocess_one_edf(str(edf_file), str(out_fif), run_ica=False)
                epochs_list.append(ep)
            except Exception as e:
                print(f"Skip {edf_file.name}: {e}")

    if epochs_list:
        combined_epochs = mne.concatenate_epochs(epochs_list)

        by_subject[subject_id] = combined_epochs

        # Save combined file
        combined_out = processed_root / subject_id / f"{subject_id}-allruns-epo.fif"
        combined_epochs.save(str(combined_out), overwrite=True)

        print(subject_id, "done.",
            {k: len(combined_epochs[k]) for k in combined_epochs.event_id})
