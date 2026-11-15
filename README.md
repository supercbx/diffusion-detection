

## Installation

Use Python 3.11 or later. Install a mutually compatible PyTorch and torchvision pair for your hardware, then install the project dependencies.



Full training loads OpenAI CLIP ViT-B/16 and two ImageNet Swin-Small encoders through timm. Weight downloads require network access or a populated cache. Optional CuPy preprocessing uses `--gpu-processor`. Install the CuPy package matching your CUDA runtime separately.

## Data and manifests

CSV manifests contain `path,label,subset,split`. Labels are 0 for real and 1 for generated. Relative paths are resolved against the CSV directory. Supported splits are `train`, `val`, `test`, and `calibration`.

Edit `configs/data_spec.example.json` to point to your GenImage directories. The example samples 10,000 real and 10,000 generated training images from each of five generators, and preserves all eight official validation subsets.

```bash
python prepare_data.py --spec configs/data_spec.example.json --output-dir manifests --train-per-class 50000 --seed 42
```

An explicit `val` section preserves the complete supplied validation split. The final manuscript uses this split for checkpoint selection and reports GenImage validation performance. For older specifications without `val`, the existing source-training holdout behavior remains available. Cross-dataset evaluation manifests should use `test`.

D3 is collected sequentially from the `train` split of `elsaEU/ELSA_D3`, targeting 10,000 generated images per generator and 40,000 real images. The supplied archive contains no original collection manifest. Prepare the D3 CSV from your saved collection metadata and identify the dataset-provided validation split explicitly. The original downloaded files and their historical ordering cannot be reconstructed from model code alone.

## Training and evaluation

```bash
python main.py train --train-manifest manifests/train.csv --val-manifest manifests/val.csv --config configs/fpra.json --output-dir runs/full --device cuda
python main.py evaluate --checkpoint runs/full/best.pt --manifest manifests/val.csv --split val --output-dir outputs/genimage --device cuda
python main.py evaluate --checkpoint runs/full/best.pt --manifest aigibench.csv --split test --output-dir outputs/aigibench --device cuda
```



```bash
python main.py train --train-manifest manifests/train.csv --val-manifest manifests/val.csv --config configs/fpra.json --resume runs/full/last.pt --output-dir runs/resumed --device cuda
```

Configuration and manifests must match to resume. Earlier AUC-selected checkpoints can be evaluated; resuming requires an accuracy-selected checkpoint so selection metrics are not mixed.

## Model and preprocessing

Shared training augmentation uses bicubic 224 by 224 resizing, horizontal flipping with probability 0.5, and JPEG compression with probability 0.4 and integer quality 60 through 100. CLIP receives normalized RGB. FFT preprocessing receives grayscale values in [0,1]. The forward FFT is unnormalized and centered, and the inverse includes 1/(HW).

Amplitude processing uses multiplicative Gaussian perturbations, a radial mask, positive-factor rejection sampling, and a 3 by 3 median filter. Phase dispersion uses ordinary finite differences on principal phases, a 3 by 3 local variance window, and reflective boundaries. Phase noise is independent across coefficients. Reconstruction takes the inverse FFT real part and applies a 5 by 5 Gaussian with sigma 1. Residuals retain their raw absolute float32 scale. 

CLIP stays frozen and in evaluation mode. The two Swin encoders have separate weights. RGB supplies one query token, and the residual branches supply two key/value tokens, with dimension 768 and single-head attention. A channel-wise sigmoid gate combines the semantic and attended features.



