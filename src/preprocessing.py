import numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, exponential_moving_standardize, create_windows_from_events

def keep_hand_classes(dataset):
    for ds in dataset.datasets:
        raw = ds.raw
        annotations = raw.annotations
        keep_mask = [('hand' in desc.lower()) for desc in annotations.description]
        raw.set_annotations(annotations[keep_mask])
    return dataset

def get_subject_data(dataset_name, subject_id, low_cut, high_cut, tmin, tmax):
    dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject_id])
    
    preprocessors = [
        Preprocessor('pick', picks='eeg'),
        Preprocessor('filter', l_freq=low_cut, h_freq=high_cut),
        Preprocessor(lambda x: exponential_moving_standardize(x, factor_new=1e-3, init_block_size=1000))
    ]
    
    preprocess(dataset, preprocessors)
    dataset = keep_hand_classes(dataset)
    sfreq = dataset.datasets[0].raw.info["sfreq"]

    windows_dataset = create_windows_from_events(
        dataset, 
        trial_start_offset_samples=int(tmin * sfreq),
        trial_stop_offset_samples=int(tmax * sfreq), 
        preload=True
    )
    
    X = np.transpose(np.array([x[0] for x in windows_dataset]), (0, 2, 1))
    y = np.array([x[1] for x in windows_dataset])
    
    return X, y