import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold

from architecture import EMPTModel
from preprocessing import get_subject_data

# --- CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DATASET_NAME = "BNCI2014_001"
LOW_CUT, HIGH_CUT = 8.0, 30.0
TMIN, TMAX = -0.5, 0
D_MODEL, NUM_HEADS = 128, 8
NUM_EXPERTS, TOP_K = 8, 4
BATCH_SIZE, EPOCHS = 32, 300
LEARNING_RATE = 5e-5
N_SPLITS, PATIENCE = 5, 15
LOG_FILE = "resultados_moe.txt"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Sujeito,Config,Fold,Best_Val_Acc\n")

all_results = {}

for current_subject in SUBJECT_IDS:
    print(f"\n{'='*40}\nPROCESSING SUBJECT: {current_subject}\n{'='*40}")
    X_subj, y_subj = get_subject_data(DATASET_NAME, current_subject, LOW_CUT, HIGH_CUT, TMIN, TMAX)
    n_classes = len(np.unique(y_subj))

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    subject_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_subj)):
        X_train, X_val = X_subj[train_idx], X_subj[val_idx]
        y_train, y_val = y_subj[train_idx], y_subj[val_idx]

        tf.keras.backend.clear_session()
        model = EMPTModel(d_model=D_MODEL, num_heads=NUM_HEADS, num_classes=n_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True)])

        best_val_acc = max(history.history["val_accuracy"])
        subject_accuracies.append(best_val_acc)
        
        with open(LOG_FILE, "a") as f:
            f.write(f"{current_subject},{NUM_EXPERTS}E_Top{TOP_K},{fold+1},{best_val_acc:.4f}\n")

        # --- Integrated Plotting ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label='Train')
        plt.plot(history.history["val_loss"], label='Val')
        plt.title(f"Loss S{current_subject} F{fold+1}")
        plt.legend(); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label='Train')
        plt.plot(history.history["val_accuracy"], label='Val')
        plt.title(f"Acc S{current_subject} F{fold+1}")
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"grafico_S{current_subject}_F{fold+1}_{NUM_EXPERTS}E.png")
        plt.close()

    all_results[current_subject] = np.mean(subject_accuracies)

print("\n" + "#"*40)
print(f"SUMMARY: {NUM_EXPERTS} Experts | Top-{TOP_K}")
for s_id, acc in all_results.items():
    print(f"Subject {s_id}: {acc:.4f}")
print("#"*40)