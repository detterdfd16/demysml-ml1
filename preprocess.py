
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

GLOBAL_EVENT_ID = {
    "rest": 1,
    "left_fist": 2,
    "right_fist": 3,
    "both_fists": 4,
    "both_feet": 5,
}
TARGET_SFREQ = 128

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

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="error")
    print("Original channels:", raw.ch_names)


    raw.pick_types(eeg=True)
    motor_chs = ["Fc3.","Fc1.","Fc2.","Fc4.","C3..","C1..","Cz..","C2..","C4..","Cp3.","Cp1.","Cpz.","Cp2.","Cp4."]
    raw.pick_channels(motor_chs, ordered=True) 

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, npad="auto", verbose="error")
    raw.filter(l_freq, h_freq)
    raw.set_eeg_reference("average", projection=False, verbose="error")

    print(f"channel names: {raw.ch_names}")

    # ICA
    # if run_ica:
    #     ica = mne.preprocessing.ICA(n_components=0.99,method="fastica",random_state=random_state, max_iter="auto",)
    #     ica.fit(raw, verbose="error")
    #     frontal = [ch for ch in ["Fp1", "Fp2", "AFp1", "AFp2"] if ch in raw.ch_names]
    #     if len(frontal) >= 1:
    #         eog_proxy = raw.copy().pick_channels(frontal).get_data().mean(axis=0)
    #         sources = ica.get_sources(raw).get_data()
    #         corr = np.array([np.corrcoef(sources[i], eog_proxy)[0, 1] for i in range(sources.shape[0])])
    #         bad_ics = np.where(np.abs(corr) > 0.3)[0].tolist() 
    #         ica.exclude = bad_ics
    #         raw = ica.apply(raw, verbose="error")
    #     else:
    #         pass

    # Events / epochs
    # Events / epochs
    events, event_id = mne.events_from_annotations(raw)

    labels = {"T0", "T1", "T2"}
    if not labels.issubset(event_id.keys()):
        raise RuntimeError(f"[{edf_path.name}] Missing {labels - set(event_id.keys())}. Found: {list(event_id.keys())}")

    # Build a per-file mapping from raw annotation codes -> global 5-class codes
    t0_code = event_id["T0"]
    t1_code = event_id["T1"]
    t2_code = event_id["T2"]

    code_map = {t0_code: GLOBAL_EVENT_ID["rest"]}

    if run in lrRun:
        code_map[t1_code] = GLOBAL_EVENT_ID["left_fist"]
        code_map[t2_code] = GLOBAL_EVENT_ID["right_fist"]
    elif run in bothRun:
        code_map[t1_code] = GLOBAL_EVENT_ID["both_fists"]
        code_map[t2_code] = GLOBAL_EVENT_ID["both_feet"]
    else:
        raise ValueError(f"{edf_path}: run R{run:02d} not in task runs (3–14).")

    # Apply remapping to the events array (3rd column is event code)
    events_remap = events.copy()
    for old_code, new_code in code_map.items():
        events_remap[events_remap[:, 2] == old_code, 2] = new_code

    # Create epochs using the GLOBAL_EVENT_ID (unique codes)
    # Only include labels that exist in THIS run (prevents "No matching events" errors)
    present_codes = set(events_remap[:, 2])
    event_id_this_run = {k: v for k, v in GLOBAL_EVENT_ID.items() if v in present_codes}

    epochs = mne.Epochs(
        raw,
        events_remap,
        event_id=event_id_this_run,   # <-- subset mapping
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="error",
    )

    # Force the global mapping for consistency across runs/subjects
    epochs.event_id = GLOBAL_EVENT_ID.copy()

    epochs.save(out_fif_path, overwrite=True)
    return epochs


# Single run test
# edf_path = r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\files\S001\S001R04.edf"
# out_fif  = r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\ml1\demysml-ml1\processed\S001R04-epo.fif"
# epochs = preprocess_one_edf(edf_path, out_fif, run_ica=True)

# print("sfreq:", epochs.info["sfreq"])
# print("event_id:", epochs.event_id)
# assert epochs.info["sfreq"] == TARGET_SFREQ, "sfreq is not 128!"
# assert epochs.event_id == GLOBAL_EVENT_ID, "event_id does not match global 5-class mapping!"
# print("✅ Single-file check passed.")

from collections import Counter

SUBJECT_ID = "S001"
root_dir = pathlib.Path(r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\files")
subject_dir = root_dir / SUBJECT_ID

task_runs = list(range(3, 15))

epochs_list = []
sfreqs = []

for run in task_runs:
    edf_file = subject_dir / f"{SUBJECT_ID}R{run:02d}.edf"
    if not edf_file.exists():
        print("Missing:", edf_file.name)
        continue

    out_fif = pathlib.Path(r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\ml1\demysml-ml1\processed_test") \
              / SUBJECT_ID / f"{edf_file.stem}-epo.fif"

    try:
        ep = preprocess_one_edf(str(edf_file), str(out_fif), run_ica=False)
        epochs_list.append(ep)
        sfreqs.append(ep.info["sfreq"])

        # Count event codes directly (robust even if label not in this run)
        code_counts = Counter(ep.events[:, 2])
        print(f"{edf_file.name} sfreq={ep.info['sfreq']}Hz codes={dict(code_counts)}")

    except Exception as e:
        print("Skip", edf_file.name, ":", e)

assert len(epochs_list) > 0, "No runs loaded."

# Check all runs are at 128 Hz
print("\nUnique sfreqs across runs:", sorted(set(sfreqs)))
assert all(s == TARGET_SFREQ for s in sfreqs), f"Not all runs resampled to {TARGET_SFREQ} Hz."

# Concatenate
epochs_all = mne.concatenate_epochs(epochs_list)

# Enforce global event_id (some runs might have subset in Epochs constructor)
epochs_all.event_id = GLOBAL_EVENT_ID.copy()

print("\nConcatenated:", epochs_all)
print("event_id:", epochs_all.event_id)

# Check event_id is exactly as desired
assert epochs_all.event_id == GLOBAL_EVENT_ID, "Global event_id mismatch after concat!"

# Check event codes present in combined data
combined_code_counts = Counter(epochs_all.events[:, 2])
print("Combined code counts:", dict(combined_code_counts))

# These should exist across all runs for the 5-class dataset
expected_codes = set(GLOBAL_EVENT_ID.values())  # {1,2,3,4,5}
missing_codes = expected_codes - set(combined_code_counts.keys())
if missing_codes:
    print("WARNING: Some classes missing in this subject:", missing_codes)
else:
    print("✅ All 5 class codes present for", SUBJECT_ID)

print("✅ One-subject test passed.")

# root_dir = pathlib.Path(r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\files")
# processed_root = pathlib.Path(r"C:\Users\Asus\OneDrive - Imperial College London\I-Explore\ml1\demysml-ml1\processed_all")
# processed_root.mkdir(parents=True, exist_ok=True)

# by_subject = {}

# for subject_dir in sorted(root_dir.glob("S*")):
#     subject_id = subject_dir.name  # "S001"
#     epochs_list = []

#     for edf_file in sorted(subject_dir.glob(f"{subject_id}R*.edf")):
#         run = extract_run_number(str(edf_file))
#         if 3 <= run <= 14:
#             out_dir = processed_root / subject_id
#             out_dir.mkdir(parents=True, exist_ok=True)
#             out_fif = out_dir / f"{edf_file.stem}-epo.fif"

#             try:
#                 ep = preprocess_one_edf(str(edf_file), str(out_fif), run_ica=False)
#                 epochs_list.append(ep)
#             except Exception as e:
#                 print(f"Skip {edf_file.name}: {e}")

#     if epochs_list:
#         combined_epochs = mne.concatenate_epochs(epochs_list)

#         by_subject[subject_id] = combined_epochs

#         # Save combined file
#         combined_out = processed_root / subject_id / f"{subject_id}-allruns-epo.fif"
#         combined_epochs.save(str(combined_out), overwrite=True)

#         print(subject_id, "done.",
#             {k: len(combined_epochs[k]) for k in combined_epochs.event_id})
