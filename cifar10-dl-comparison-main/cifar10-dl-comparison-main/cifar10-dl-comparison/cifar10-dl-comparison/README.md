# Comparative Analysis of Deep Learning Architectures for CIFAR-10

ICT 4442 Deep Learning Mini Project — School of Computer Engineering, MIT Manipal.

The project trains four models from four different architecture families on CIFAR-10. All four use the same split, the same augmentation, the same epoch budget and the same evaluation script, so the differences in the results come from the architecture.

| Model | Family | File | Owner |
|---|---|---|---|
| MLP baseline | Fully connected | `src/models/mlp.py` | Harsh Kushwaha (230953366) |
| ResNet-style CNN | Convolutional | `src/models/cnn.py` | Sanket Tippimani (230911200) |
| RNN/LSTM | Recurrent | `src/models/lstm.py` | Sanyam Satija (230911204) |
| Vision Transformer | Attention | `src/models/vit.py` | Devansh Sharma (230911382) |

## How each model "sees" the same image

Every model receives the same normalised `3 x 32 x 32` tensor. What differs is how it reads that tensor, and this is the inductive bias the project compares.

| Model | Input as the model sees it | Built-in assumption (inductive bias) | Expected behaviour |
|---|---|---|---|
| MLP | One flat vector of 3,072 numbers | None. Neighbouring pixels are treated like any other pair | Lowest accuracy; over-fits fastest |
| CNN | 2-D grid scanned by shared 3x3 filters | Locality and translation equivariance | Highest accuracy for this data size |
| LSTM | Sequence of 32 rows (96 values each), or 64 patches (48 values each) | Order matters; information flows step by step | Below CNN; vertical structure is only carried through memory |
| ViT | 64 patch tokens of size 4x4 plus a [CLS] token | Weak prior; global attention between all patches from layer 1 | Data-hungry from scratch; gap to CNN depends on augmentation and epochs |

Parameter counts with the default configs (printed by `smoke_test.py`) are approximately MLP 3.81M, CNN 2.78M, LSTM 0.89M and ViT 1.81M. Report these next to accuracy in the final table.

## Repository structure

```
cifar10-dl-comparison/
├── configs/
│   ├── common.yaml        # shared protocol (seed, split, augmentation, epochs) - same for all
│   ├── mlp.yaml  cnn.yaml  lstm.yaml  vit.yaml   # per-model architecture + optimiser
├── src/
│   ├── data.py            # CIFAR-10 download, 45k/5k/10k split, transforms
│   ├── engine.py          # train / evaluate loops (AMP, grad clipping, scheduler)
│   ├── metrics.py         # accuracy, macro P/R/F1, confusion matrix, plots
│   ├── utils.py           # config merge, seeding, optimiser and LR schedule
│   └── models/            # one file per member + registry in __init__.py
├── train.py               # train one model, keep best checkpoint by val accuracy
├── evaluate.py            # test-set metrics for one model (run once at the end)
├── compare.py             # single comparison table across all models
├── error_analysis.py      # most-confused class pairs + most confident mistakes
├── smoke_test.py          # 10-second check, no download needed
└── run_all.sh             # train + evaluate + analyse all four, then compare
```

## Setup

```bash
git clone <repo-url> && cd cifar10-dl-comparison
pip install -r requirements.txt
python smoke_test.py            # run this first: all four models should print OK
```

**Google Colab (free GPU):** Runtime → Change runtime type → GPU. Then run:

```python
!git clone <repo-url>
%cd cifar10-dl-comparison
!pip install -q pyyaml
!python smoke_test.py
!python train.py --model cnn
```

Colab deletes files when the session ends. Copy `results/` to Google Drive after each run.

## Usage

```bash
# 1. Train (CIFAR-10 downloads automatically to ./data on first run)
python train.py --model mlp
python train.py --model cnn
python train.py --model lstm
python train.py --model vit

# 2. Evaluate on the test set (do this only for the final model of each member)
python evaluate.py --model cnn

# 3. Error analysis for any model
python error_analysis.py --model cnn

# 4. Comparison table across every model that has been evaluated
python compare.py

# Or everything in one go
./run_all.sh
```

**Changing settings without editing files.** Use `--set` to override any config value:

```bash
python train.py --model vit  --set optim.lr=0.0005 --set model.depth=8 --run-name vit_depth8
python train.py --model lstm --set model.sequence=patches --run-name lstm_patches
python train.py --model lstm --set model.bidirectional=true --run-name lstm_bi
```

`--run-name` saves to a separate folder, so tuning runs don't overwrite the main result. Only the folders named `mlp`, `cnn`, `lstm` and `vit` go into `compare.py`.

**Quick pipeline check without downloading data:**

```bash
python train.py --model mlp --fake-data --epochs 2 --max-batches 5
python evaluate.py --model mlp --fake-data
```

Remember to delete `results/mlp` afterwards, because these runs use random images.

## Common evaluation protocol

| Setting | Value (from `configs/common.yaml`) |
|---|---|
| Split | 45,000 train / 5,000 val (stratified, seed 42) / 10,000 official test |
| Normalisation | per-channel mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616) |
| Augmentation (train only) | RandomCrop 32 with padding 4, horizontal flip; optional Cutout for all models |
| Budget | 50 epochs, batch size 128, best checkpoint chosen on validation accuracy |
| Metrics | accuracy, macro precision / recall / F1, confusion matrix, per-class F1 |

The optimiser, learning rate and schedule are set per model, because each family has its own standard training recipe. Change shared settings only in `common.yaml`, and agree on the change as a team.

## Outputs

| File | Produced by | Used for |
|---|---|---|
| `results/<model>/history.csv`, `training_curves.png` | train.py | over-fitting discussion |
| `results/<model>/best.pt`, `config_used.json`, `train_summary.json` | train.py | reproducibility |
| `results/<model>/test_metrics.json`, `confusion_matrix.png` | evaluate.py | results section |
| `results/<model>/confused_pairs.csv`, `misclassified.png` | error_analysis.py | error analysis |
| `results/comparison.md/.csv/.png`, `per_class_f1.png` | compare.py | single comparison table |

## Tuning notes per model

- **MLP:** more dropout or weight decay reduces over-fitting. Expect the train/val gap to be the largest of the four models.
- **CNN:** SGD with lr 0.1 and cosine decay is the standard ResNet recipe. Width (`channels`) matters more than depth at this budget.
- **LSTM:** keep gradient clipping on. Compare `sequence: rows` against `patches`, and try `bidirectional: true`; these make useful ablations for the report.
- **ViT:** the warm-up is important. If accuracy plateaus low, try more epochs (`--set train.epochs=100`), label smoothing 0.1, or `patch_size: 2`. Note that patch size 2 gives 256 tokens and is about 4x slower. Report any change that breaks the common protocol as a separate row.

## Git workflow (per-member commits are graded)

Commit history is used to verify individual contributions, so each member commits their own work from their own account:

```bash
git checkout -b feat/vit            # one branch per member/model
# ...edit src/models/vit.py, configs/vit.yaml...
git add src/models/vit.py configs/vit.yaml
git commit -m "vit: add warm-up and label smoothing experiment"
git push origin feat/vit            # then open a pull request into main
```

Commit small and often. Do not upload `data/` or `.pt` checkpoints; `.gitignore` already excludes them.

## Academic integrity

Per Section 9 of the course guidelines, any use of external code or LLMs must be cited, with the exact place it was used. Record here which files were scaffolded with an AI assistant and what each member changed or wrote themselves.
