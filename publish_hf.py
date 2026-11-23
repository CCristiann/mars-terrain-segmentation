"""Publish the final model and the browser demo to the Hugging Face Hub.

    uv run --with onnx --with onnxscript python publish_hf.py                # model + Space
    uv run --with onnx --with onnxscript python publish_hf.py --only model   # weights, ONNX, card
    uv run python publish_hf.py --only space                                 # static demo (hf/space/)

Add --export-only to write outputs/hf_export without uploading.
onnx/onnxscript are only needed for the ONNX export, so they are not project dependencies.
Needs a write token: `uv run hf auth login`.
"""
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi

from augmentation import MEAN, STD
from utils import load_trained_model

USER = "ccristiann"
MODEL_NAME = "mars-terrain-unet-resnet34"
SPACE_NAME = "mars-terrain-segmentation"


class BrowserModel(nn.Module):
    """The whole pipeline in one graph, so the web page only passes raw pixels.

    in:  uint8 RGB image (H, W, 3), any size
    out: uint8 class map (H, W), 0 soil, 1 bedrock, 2 sand, 3 big_rock
    Same steps as evaluate_test.py: normalize, resize to img_size, predict, resize back.
    """

    def __init__(self, model, size):
        super().__init__()
        self.model, self.size = model, size
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, image):
        h, w = image.shape[0], image.shape[1]
        x = (image.permute(2, 0, 1)[None].float() - self.mean) / self.std
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        logits = F.interpolate(self.model(x), size=(h, w), mode="bilinear", align_corners=False)
        return logits.argmax(1)[0].to(torch.uint8)


def weights_to_fp16(path):
    """Store the big weights in 16 bits (half the download), computed in 32 bits.

    Each weight gets a Cast node back to float32, so the math is unchanged and every
    ONNX Runtime backend can run it; predictions match the fp32 model on ~99.99% of pixels.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(path)
    g = m.graph
    casts = []
    for init in list(g.initializer):
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.size > 1024:
            g.initializer.remove(init)
            g.initializer.append(numpy_helper.from_array(a.astype(np.float16), init.name + "_fp16"))
            casts.append(helper.make_node("Cast", [init.name + "_fp16"], [init.name], to=TensorProto.FLOAT))
    for node in reversed(casts):
        g.node.insert(0, node)
    onnx.checker.check_model(m)
    onnx.save(m, path)


def export_onnx(model, size, path):
    dim = lambda name: torch.export.Dim(name, min=64, max=4096)  # noqa: E731
    example = torch.randint(0, 256, (1024, 1024, 3), dtype=torch.uint8)
    torch.onnx.export(
        BrowserModel(model, size).eval(),
        (example,),
        path,
        input_names=["image"],
        output_names=["classes"],
        dynamic_shapes={"image": {0: dim("h"), 1: dim("w")}},
        opset_version=18,
        external_data=False,  # one self-contained file for the browser
    )
    weights_to_fp16(path)


def export_model(run_dir, out_dir):
    """model.safetensors + config.json (for smp.from_pretrained), model.onnx (for the demo)."""
    model, cfg = load_trained_model(run_dir, torch.device("cpu"))
    model.save_pretrained(out_dir)  # config.json tells smp how to rebuild the network
    shutil.copy("hf/model_card.md", out_dir / "README.md")  # replace the auto-generated card
    export_onnx(model, cfg["img_size"], out_dir / "model.onnx")


def publish_model(api, repo_id, out_dir):
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=out_dir, commit_message="Upload model")
    print("model ->", f"https://huggingface.co/{repo_id}")


def publish_space(api, repo_id):
    # static Space: plain HTML/JS, free to host; inference runs in the visitor's browser
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path="hf/space",
        ignore_patterns=[".DS_Store", "*.onnx", "__pycache__/*"],
        delete_patterns=["style.css", "__pycache__/*"],  # HF template leftovers
        commit_message="Update demo",
    )
    print("space ->", f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=USER, help="Hugging Face username")
    ap.add_argument("--run", default="outputs/gpu_long", help="run folder with best.pt")
    ap.add_argument("--only", choices=["model", "space"])
    ap.add_argument("--export-only", action="store_true", help="write outputs/hf_export, no upload")
    args = ap.parse_args()

    out_dir = Path("outputs/hf_export")
    if args.only != "space":
        export_model(args.run, out_dir)
        print("exported ->", out_dir)
    if args.export_only:
        raise SystemExit

    api = HfApi()
    if args.only != "space":
        publish_model(api, f"{args.user}/{MODEL_NAME}", out_dir)
    if args.only != "model":
        publish_space(api, f"{args.user}/{SPACE_NAME}")
