---
license: mit
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
tags:
  - semantic-segmentation
  - unet
  - resnet34
  - mars
  - planetary-science
  - pytorch
metrics:
  - mean_iou
---

# Mars Terrain Segmentation — U-Net / ResNet34

Semantic segmentation of Martian terrain from Curiosity NavCam images into
**soil, bedrock, sand, big rock**. Trained on NASA's **AI4Mars** dataset.

- 🔗 Code: https://github.com/CCristiann/mars-terrain-segmentation
- 🚀 Demo: https://huggingface.co/spaces/ccristiann/mars-terrain-segmentation

## Results (expert-labelled test set, 322 images)

| Test variant | soil | bedrock | sand | big rock | **mIoU** |
|---|---|---|---|---|---|
| min3 (all 3 experts agree) | 0.981 | 0.929 | 0.958 | 0.372 | **0.810** |
| min1 (≥1 expert label) | 0.915 | 0.805 | 0.851 | 0.109 | **0.670** |

Big rock has high recall (0.997 on min3) but low precision (0.37): the model
finds rocks but over-predicts them.

## Usage

```python
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

model = smp.from_pretrained("ccristiann/mars-terrain-unet-resnet34").eval()

# x: float tensor (1, 3, H, W), RGB in [0, 1] (grayscale NavCam repeated on 3 channels)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x = F.interpolate((x - mean) / std, size=(512, 512), mode="bilinear", align_corners=False)
with torch.no_grad():
    pred = model(x).argmax(1)  # 0 soil, 1 bedrock, 2 sand, 3 big_rock
```

## Training

- Architecture: U-Net, ResNet34 encoder pre-trained on ImageNet, input 512×512
- Data: AI4Mars MSL NavCam, train/val split grouped by acquisition time (no leakage)
- Loss: CrossEntropy (class weights 0.369 / 0.317 / 0.7 / 1.0) + 0.5 · Dice
- 30 epochs, AdamW lr 3e-4, batch 8, bf16 mixed precision, RTX 4090
- Loss ablation (Tversky, Lovász, focal): lowering the big-rock weight helped more
  than changing the loss — details in the GitHub README.

## Limitations

- Trained only on Curiosity **NavCam** images; other cameras or rovers are out of distribution.
- Unlabelled regions (rover hardware, sky, far terrain) were ignored in training,
  but the model still assigns a class to every pixel there.
- Big rock is the weak class (low precision).

## Files

- `model.safetensors` + `config.json`: PyTorch weights, load with `smp.from_pretrained`.
- `model.onnx` (49 MB): the full pipeline for the browser demo. Input: uint8 RGB image `(H, W, 3)`
  of any size; output: uint8 class map `(H, W)`. Normalisation and resizing are inside the graph;
  weights are stored in fp16 and computed in fp32 (matches PyTorch on ~99.99% of pixels).

## License and attribution

- **These weights:** MIT.
- **Training data:** [AI4Mars v0.6](https://doi.org/10.5281/zenodo.15995036) by R. Swan, M. Ono, D. Atha et al.
  (NASA JPL) and the Zooniverse volunteer labelers, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The model is trained on this data; the data
  itself is not included in this repository.
- **Encoder initialisation:** torchvision ResNet34 ImageNet weights (BSD-3-Clause). ImageNet itself has
  non-commercial terms of access: check them before any commercial use.

## Citation

If you use this model, please cite the dataset:

```bibtex
@inproceedings{swan2021ai4mars,
  title     = {AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars},
  author    = {Swan, R. Michael and Atha, Deegan and Leopold, Henry A. and Gildner, Matthew and
               Oij, Stephanie and Chiu, Cindy and Ono, Masahiro},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2021},
  pages     = {1982--1991},
  doi       = {10.1109/CVPRW53098.2021.00226}
}
```
