# Reproducing the EMPT Pipeline: An Open-Source Approach to Motor-Imagery EEG Classification

This repository provides the implementation of the **EMPT** pipeline (EEG Mixture-of-Experts ProbSparse Transformer).

The project focuses on the classification of electroencephalographic (EEG) signals for **Motor Imagery (MI)**, using a Transformer-based architecture with efficiency and specialization mechanisms.

## Project Summary

This work reproduces and organizes an end-to-end processing pipeline, from raw data acquisition to final classification. The EMPT model's key contribution lies in its combination of:

- **ProbSparse Attention**: A variant of classical attention that reduces computational complexity by focusing on the query-key pairs with the highest information sparsity.
- **Mixture-of-Experts (MoE)**: A layer that activates only a subgroup of "experts" (dense neural networks) for each sample, enabling greater learning capacity without a prohibitive computational cost.

## System Requirements

The pipeline was developed and tested using **Python 3.8+**. To ensure reproducibility of results, the versions of the main libraries are:

* **TensorFlow**: `2.15.1`
* **Braindecode**: `0.8.1`
* **MNE**: `1.8.0`
* **MOABB**: `1.2.0`
* **Scikit-learn**: `1.5.2`
* **NumPy**: `1.26.4`
* **Matplotlib**: `3.9.4`

More details are available in `requirements.txt`.

## File Structure

The code was modularized to follow software engineering good practices:

* `architecture.py`: Contains all custom layers (`TopKMoE`, `ProbSparseAttention`) and the model structure (`EMPTModel`).
* `preprocessing.py`: Functions for automatically loading datasets via MOABB and preprocessing via Braindecode (filtering, exponential normalization and windowing).
* `main.py`: Entry point of the experiment. Coordinates K-Fold, training and graph generation.

## Technologies Used

* **TensorFlow/Keras**: Main Deep Learning framework.
* **MNE-Python**: Neurophysiological data processing.
* **Braindecode**: Library specialized in Deep Learning for EEG.
* **MOABB**: Mother of All BCI Benchmarks, used to guarantee dataset reproducibility.

## How to Use

### 1. Installing Dependencies

```bash
pip install -r requirements.txt
```

or:

```bash
pip install tensorflow==2.15.1 braindecode==0.8.1 moabb==1.2.0 mne==1.8.0 scikit-learn==1.5.2 matplotlib==3.9.4
```

### 2. Execution

To start training and K-Fold cross-validation:

```bash
python main.py
```

## Results and Logs

The pipeline automatically generates:

* `resultados_moe.txt`: A detailed log containing the accuracy of each fold per subject.
* **Training plots**: `.png` files with Loss and Accuracy curves for each fold of the cross-validation.

## Academic Context

This repository originated as part of an undergraduate thesis (*Trabalho de Conclusão de Curso*) in Computer Engineering at the Federal University of Pelotas (UFPel), Brazil, under the supervision of Prof. Dr. Marilton Sanchotene de Aguiar (CDTec/UFPel). It has since been extended and published as a research paper at **BRACIS 2026**.

## Citation

If you use this code in your research, please cite the associated paper:

```bibtex
@inproceedings{martinez2026empt,
  author    = {Martinez, Santiago Del Valle Alvarez and de Aguiar, Marilton Sanchotene},
  title     = {Reproducing the EMPT Pipeline: An Open-Source Approach to Motor-Imagery EEG Classification},
  booktitle = {Proceedings of the 15th Brazilian Conference on Intelligent Systems (BRACIS)},
  year      = {2026},
  address   = {Brazil}
}
```

If you wish to cite the original undergraduate thesis specifically:

```bibtex
@thesis{martinez2026empt_tcc,
  author  = {Martinez, Santiago Del Valle Alvarez and de Aguiar, Marilton Sanchotene},
  title   = {Reproducing EMPT: An Open-Source Pipeline for Motor-Imagery EEG Classification},
  school  = {Universidade Federal de Pelotas},
  year    = {2026},
  type    = {Undergraduate Thesis},
  address = {Pelotas, Brazil}
}
```

---

*This repository is provided for research and reproducibility purposes in the field of Brain-Computer Interfaces (BCI).*
