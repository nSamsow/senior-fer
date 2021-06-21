from math import ceil, floor

import cv2
import onnx
import onnxruntime
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from models.models import (nfnet_l0, ran_nfnet_l0, ran_resnet18, ran_resnet50,
                           resnet18, resnet50, tan_nfnet_l0, tan_resnet18,
                           tan_resnet50)
from PIL import Image
from timm.models.pnasnet import FactorizedReduction

onnx_model = onnx.load("../onnx_models/r18.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("../onnx_models/r18.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


batch_size = 1
size = 224
img = torch.randn(batch_size, 3, size, size, requires_grad=True)

# Transforms
resize = transforms.Resize([size, size])
img = resize(img)

to_tensor = transforms.ToTensor()
img = to_tensor(img)
img.unsqueeze_(0)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
ort_outs = ort_session.run(None, ort_inputs)

out = ort_outs[0]


def main():
    # face_detector =

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

    if "l0" in args.model:
        size = 288
    else:
        size = 224

    batch_size = 1

    onnx_model = onnx.load("../onnx_models/r18.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("../onnx_models/r18.onnx")

    label_map = {
        0: "Neutral",
        1: "Happiness",
        2: "Surprise",
        3: "Sadness",
        4: "Anger",
        5: "Disgust",
        6: "Fear",
        7: "Contempt",
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScaleFPS = 0.5
    emoThick, fpsThick, bboxThick = 2, 2, 1
    color = (0, 255, 0)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bboxes, det_time = face_detector.predict(
            frame,
            resize_scale=1,
            score_threshold=0.6,
            top_k=10000,
            NMS_threshold=0.4,
            NMS_flag=True,
            skip_scale_branch_list=[],
        )

        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2, _ = bbox
                x1, y1, x2, y2 = floor(x1), floor(y1), ceil(x2), ceil(y2)

                face = frame[y1:y2, x1:x2]

                face = TF.resize(face, [size, size])
                face = TF.rgb_to_grayscale(face, num_output_channels=3)
                face = TF.normalize(face, [131.0912, 103.8827, 91.4953], [255, 255, 255])
                face = TF.to_tensor(face)
                face.unsqueeze_(0)

                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(face)}
                ort_outs = ort_session.run(None, ort_inputs)
                output = ort_outs[0]

                # output = label_map[]
