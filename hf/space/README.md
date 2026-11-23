---
title: Mars Terrain Segmentation
emoji: 🪐
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - ccristiann/mars-terrain-unet-resnet34
short_description: Segment Martian terrain in Curiosity NavCam images
---

U-Net (ResNet34) trained on NASA AI4Mars to segment Curiosity NavCam images into
soil, bedrock, sand and big rock. The model (ONNX, 49 MB) runs in the browser with
ONNX Runtime Web: images are never uploaded.

Code: https://github.com/CCristiann/mars-terrain-segmentation

## Licenses

- Demo code and model weights: MIT.
- Example images (`examples/`, unmodified): from [AI4Mars v0.6](https://doi.org/10.5281/zenodo.15995036)
  by R. Swan, M. Ono, D. Atha et al. (NASA JPL) and the Zooniverse volunteer labelers,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The coloured overlays shown in the demo are
  model predictions computed on these images.
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime): MIT. Fonts (Google Fonts): SIL OFL 1.1.
