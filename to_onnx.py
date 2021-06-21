import argparse

import torch.onnx
import torch.utils.model_zoo as model_zoo

from models.models import (nfnet_l0, ran_nfnet_l0, ran_resnet18, ran_resnet50,
                           resnet18, resnet50, tan_nfnet_l0, tan_resnet18,
                           tan_resnet50)

parser = argparse.ArgumentParser(description="PyTorch-to-ONNX conversion")
parser.add_argument(
    "--model",
    metavar="",
    default="nfnet_l0",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--weight-path",
    metavar="PATH",
    default="",
    type=str,
    help="Path to a model's weights",
)
parser.add_argument(
    "--export-path",
    metavar="PATH",
    default="",
    type=str,
    help="Path to the exported ONNX",
)

args = parser.parse_args()

if args.model == "resnet18":
    torch_model = resnet18()
elif args.model == "resnet50":
    torch_model = resnet50()
elif args.model == "nfnet_l0":
    torch_model = nfnet_l0()
elif args.model == "ran_resnet18":
    torch_model = ran_resnet18()
elif args.model == "ran_resnet50":
    torch_model = ran_resnet50()
elif args.model == "ran_nfnet_l0":
    torch_model = ran_nfnet_l0()
elif args.model == "tan_resnet18":
    torch_model = tan_resnet18()
elif args.model == "tan_resnet50":
    torch_model = tan_resnet50()
elif args.model == "tan_nfnet_l0":
    torch_model = tan_nfnet_l0()

torch_model.load_state_dict(torch.load(args.weight_path)["state_dict"])
torch_model.eval()

size = 288 if "l0" in args.model else 224
batch_size = 1


if any(x in args.model for x in ["ran", "tan"]):
    x = torch.randn(batch_size, 3, size, size, 6, requires_grad=True)
else:
    x = torch.randn(batch_size, 3, size, size, requires_grad=True)

torch_out = torch_model(x)

torch.onnx.export(
    torch_model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    args.export_path,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=13,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },  # variable length axes
)
