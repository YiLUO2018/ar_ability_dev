"""导出 PyTorch 模型为 TFLite（litert-torch，即原 ai-edge-torch）。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from models import CornerModel, CornerModelExport, PieceClassifier, PieceClassifierExport


def export_corner(checkpoint, out_path, size):
    import litert_torch as lt

    model = CornerModel(input_size=size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    wrapper = CornerModelExport(model).eval()
    x = torch.rand(1, size, size, 3)  # NHWC
    edge = lt.convert(wrapper, (x,))
    edge.export(out_path)


def export_classify(checkpoint, out_path):
    import litert_torch as lt

    model = PieceClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    wrapper = PieceClassifierExport(model).eval()
    x = torch.rand(1, 400, 360, 3)  # NHWC
    edge = lt.convert(wrapper, (x,))
    edge.export(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["corner", "classify"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    if args.model == "corner":
        export_corner(args.checkpoint, args.out, args.size)
    else:
        export_classify(args.checkpoint, args.out)
    print("exported →", args.out)


if __name__ == "__main__":
    main()
