# Image Classification with CNNs on a CPU Budget

Experiments for a diploma thesis (University of Lodz).

**Research question:** how do architecture depth, kernel size / stride and transfer learning affect accuracy and training cost when training on a CPU?

## Setup at a glance

| Item | Details |
|---|---|
| Datasets | Fashion-MNIST (60k train images, 10% held out for validation); CIFAR-10 (fixed random 10,000-image training subset, 10% held out for validation). Official test sets (10k images each) are used only for final evaluation. |
| Models | **baseline**: 2 conv layers (16, 32 filters) + small head. **deeper**: 3 blocks of 2 convs with batch norm, max pooling and dropout 0.25. **mobilenet_frozen**: pretrained ImageNet MobileNetV3-small, features extracted once, only a final linear layer (1024 -> 10) is trained. |
| Training | Adam, lr 1e-3, batch size 128, 10 epochs (Fashion-MNIST) / 15 epochs (CIFAR-10), CPU only. |
| Seeds | Main experiments: 5 seeds on Fashion-MNIST, 3 on CIFAR-10. Kernel/stride ablation: 3 seeds. |
| Hardware | Google Colab CPU runtime (Intel Xeon @ 2.20 GHz, 12.27 GB RAM). |
| Framework | PyTorch + torchvision, pandas, matplotlib. |

Preprocessing: CNNs get per-channel standardisation using **training-set** mean/std only. MobileNet gets greyscale repeated to 3 channels (Fashion-MNIST), bilinear resize to 96x96 and ImageNet normalisation.

## Repository contents

```
ImageClassificationCNNs.ipynb   # Full pipeline: data, models, training, analysis, figures
results.csv                     # One row per finished run (dataset, model, seed, test_acc, train_time_s, params)
curves.csv                      # Per-epoch train loss, val accuracy, val loss for every run
figures/                        # Figures for the thesis (PNG at 300 dpi + PDF) and summary_table.csv
graphs/                         # Ablation heatmap and training/validation curves (PNG)
```

### Figures

| File | Shows |
|---|---|
| `figures/fig_preprocessing_{fashion,cifar}` | Sample test images at each preprocessing step (original, standardised, MobileNet input). |
| `figures/fig_acc_vs_time_{fashion,cifar}` | Test accuracy vs training time (log axis), mean ± std with individual seeds. |
| `figures/fig_acc_vs_time_fashion_with_ablation` | Same, including the kernel/stride ablation configurations. |
| `figures/fig_confusion_*` | Row-normalised confusion matrices pooled over seeds (deeper and mobilenet_frozen). |
| `figures/summary_table.csv` | Mean/std accuracy and time, parameters, number of seeds per dataset and model. |
| `graphs/fig_ablation_fashion.png` | Heatmaps of accuracy and time over kernel size {3,5,7} x stride {1,2}. |
| `graphs/fig_curves_{fashion,cifar}.png` | Training loss and validation loss per epoch, mean ± 1 std over seeds. |

## Results summary

Mean over seeds (from `figures/summary_table.csv`). Time = training time on CPU; for MobileNet it includes one-off feature extraction.

| Dataset | Model | Test acc. (%) | Time (s) | Trainable params |
|---|---|---|---|---|
| Fashion-MNIST | baseline | 88.28 ± 0.28 | 276 | 38,282 |
| Fashion-MNIST | deeper | 91.30 ± 1.14 | 982 | 72,890 |
| Fashion-MNIST | mobilenet_frozen | 87.95 ± 0.13 | 170 | 10,250 |
| CIFAR-10 (10k subset) | baseline | 58.03 ± 1.60 | 85 | 38,570 |
| CIFAR-10 (10k subset) | deeper | 64.53 ± 0.81 | 314 | 73,178 |
| CIFAR-10 (10k subset) | mobilenet_frozen | 76.93 ± 0.08 | 46 | 10,250 |

Kernel / stride ablation (Fashion-MNIST, baseline CNN, 3 seeds): larger kernels raise accuracy (k=3: 88.3%, k=5: 89.2%, k=7: 89.9% at stride 1), but cost grows quickly (276 s to 545 s). Stride 2 cuts time by roughly 2.5-3.7x with a small accuracy change (k=5, s=2: 89.5% in 126 s; k=7, s=2: 89.8% in 148 s).

Takeaways: on the small CIFAR-10 subset, transfer learning wins on both accuracy and time. On Fashion-MNIST, the deeper CNN is most accurate but about 3.6x slower than the baseline, and the frozen MobileNet is fastest but slightly less accurate than the baseline. Note the deeper model on Fashion-MNIST has one low-accuracy seed (seed 3: 89.27%), which drives its higher std.

## How to reproduce

The notebook was run on **Google Colab (CPU runtime)** with Google Drive for storage. Timings depend on hardware, so expect accuracy to match closely but training times to differ on another machine.

### Option A: Google Colab (as originally run)

1. Upload `ImageClassificationCNNs.ipynb` to Colab and set the runtime to **CPU** (Runtime > Change runtime type). Use CPU to keep timings comparable.
2. Run all cells top to bottom. Cell 5 mounts Google Drive and writes everything under `MyDrive/thesis_results/` (results, curves, confusion matrices, predictions, figures, graphs). Datasets and cached MobileNet features go under `MyDrive/thesis_data/`.
3. Fashion-MNIST and CIFAR-10 download automatically on first use. Pretrained MobileNetV3-small weights download via torchvision.
4. If the session disconnects, rerun the notebook. `already_done()` skips every (dataset, model, seed) already logged in `results.csv`, so training resumes where it stopped.
5. Expected total runtime is several hours on a Colab CPU (the deeper model on Fashion-MNIST takes about 16 min per seed; ablation configs up to ~9 min per seed).

### Option B: run locally

1. Install dependencies (Python 3.10+):
   ```bash
   pip install torch torchvision numpy pandas matplotlib jupyter
   ```
2. In the notebook, remove or skip `from google.colab import drive` and `drive.mount(...)`, and change the paths in the setup cell to local folders, e.g.:
   ```python
   LOG_DIR  = './thesis_results'
   DATA_DIR = './thesis_data'
   ```
   (The notebook itself notes: "Pass the local paths if the notebook is running locally".) The `!lscpu` / `!free -h` cells are Linux-only and can be skipped.
3. Run all cells in order:
   ```bash
   jupyter notebook ImageClassificationCNNs.ipynb
   ```

### Regenerating figures from the provided CSVs only

You do not need to retrain to rebuild the tables and most plots. Put `results.csv` and `curves.csv` into `LOG_DIR`, run the setup cells (imports, paths, helpers), then run the analysis cells: summary table, kernel/stride heatmap, training curves, accuracy-vs-time plots. The confusion-matrix and prediction figures additionally need the `cm_*.npy` / `preds_*.npy` files, which training produces in `LOG_DIR` and which are not committed here.

### Notebook cell order

| Section | What it does |
|---|---|
| 1. Setup and logging | Imports, paths, `log_run`, `already_done`, `set_seed`. |
| 2. Data | `load_data` (fixed subset and split, selection seed 0) and `normalise` (train-set statistics). |
| Preprocessing | Preprocessing figure. |
| 3. Models | `baseline(in_ch, k, s)`, `deeper(in_ch)`, MobileNet feature extraction with on-disk caching. |
| 4. Training and evaluation | `train_run`: trains one model/seed, logs results, curves and confusion matrix. |
| Main experiments | 3 models x 5 seeds (Fashion-MNIST), x 3 seeds (CIFAR-10). |
| Kernel/stride ablation | k in {3,5,7} x s in {1,2}, Fashion-MNIST baseline, 3 seeds. |
| Predictions rerun | Retrains seed 0 of each main model to save predictions and checks that the confusion matrix matches the logged run. |
| Analysis | Example predictions, ablation heatmap, confusion matrices, curves, accuracy-vs-time, summary table, stability / timing checks. |

## Reproducibility notes

- Every run calls `set_seed(seed)` (Python, NumPy, PyTorch). The CIFAR-10 subset and the train/validation split use a fixed `RandomState(0)`, identical across models and seeds.
- Exact bit-for-bit results can still vary across PyTorch versions and CPU types. The notebook verifies reproducibility for seed 0 by comparing a rerun's confusion matrix with the logged one.
- Training time covers the training loop only (not evaluation). MobileNet time adds the one-off feature-extraction time. Timings are affected by shared Colab CPUs, so treat small time differences with caution.
- The baseline config (k=3, s=1) in the ablation is the same run as the main `baseline` model, so seeds 0-2 are not retrained.
- Pinned library versions were not recorded in the notebook. Record `pip freeze` when rerunning.

## Known limitations

- CIFAR-10 uses only 10,000 training images, so absolute CIFAR accuracies are far below full-dataset results.
- `figures/fig_confusion_cifar_deeper.png` was generated before the notebook was fixed to use `CIFAR_CLASSES` (it used the Fashion-MNIST class names), so its axis labels are wrong. The notebook code is now correct. Rerun the confusion-matrix cell to regenerate the figure; it needs the `cm_cifar_deeper_seed*.npy` files that training produces.
- Epoch counts (10 / 15) were chosen from a pilot run's validation curves. The pilot is not included.
- Single hardware setup and few seeds (3-5), so small differences between models are within noise.
