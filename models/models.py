import torch
import torch.utils.model_zoo as model_zoo
from timm.models import nfnet, resnet

from models.my_nfnet import _create_normfreenet
from models.my_resnet import My_ResNet

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet50": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
    "nfnet_l0": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nfnet_l0_ra2-45c6688d.pth",
}


def resnet18(pretrained=True, num_classes=8):
    return resnet.resnet18(pretrained, num_classes=num_classes, drop_rate=0.0)


def resnet50(pretrained=True, num_classes=8):
    return resnet.resnet50(pretrained, num_classes=num_classes, drop_rate=0.0)


def nfnet_l0(pretrained=True, num_classes=8):
    return nfnet.nfnet_l0(
        pretrained, num_classes=num_classes, drop_rate=0.5, drop_path_rate=0.25
    )


def ran_resnet18(
    pretrained=True, num_classes=8, weights_path="", strict=False, **kwargs
):
    model = My_ResNet(
        block=resnet.BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        use_cbam=False,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict)
    if weights_path:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["state_dict"], strict)

    return model


def ran_resnet50(
    pretrained=True, num_classes=8, weights_path="", strict=False, **kwargs
):
    model = My_ResNet(
        block=resnet.Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        use_cbam=False,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict)
    if weights_path:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["state_dict"], strict)

    return model


def ran_nfnet_l0(
    pretrained=True, num_classes=8, weights_path="", strict=False, **kwargs
):
    model = _create_normfreenet(
        "my_nfnet_l0",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.5,
        drop_path_rate=0.25,
        use_cbam=False,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["nfnet_l0"]), strict)
    if weights_path:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["state_dict"], strict)

    return model


def tan_resnet18(
    pretrained=True, num_classes=8, weights_path="", strict=False, **kwargs
):
    model = My_ResNet(
        block=resnet.BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        use_cbam=True,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict)
    if weights_path:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["state_dict"], strict)

    return model


def tan_resnet50(
    pretrained=True, num_classes=8, weights_path="", strict=False, **kwargs
):
    model = My_ResNet(
        block=resnet.Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        use_cbam=True,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict)
    if weights_path:
        model.load_state_dict(torch.load(weights_path), strict)

    return model


def tan_nfnet_l0(
    pretrained=True, num_classes=8, weights_path="", strict=False, **kwargs
):
    model = _create_normfreenet(
        "my_nfnet_l0",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.5,
        drop_path_rate=0.25,
        use_cbam=True,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["nfnet_l0"]), strict)
    if weights_path:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["state_dict"], strict)

    return model
