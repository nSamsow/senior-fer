import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def print_total_time(timedelta):
    """Converts 'timedelta' into 'days, hours, minutes, seconds' format

    Args:
        timedelta ([type]): [description]
    """
    time_in_sec = timedelta.total_seconds()
    days = divmod(time_in_sec, 86400)
    hours = divmod(days[1], 3600)
    minutes = divmod(hours[1], 60)
    seconds = divmod(minutes[1], 1)
    print(
        "Total time: %d days, %d hours, %d minutes and %d seconds"
        % (days[0], hours[0], minutes[0], seconds[0])
    )


def set_seed(seed):
    torch.manual_seed(seed)  # to seed the RNG for all devices (both CPU and CUDA)
    random.seed(seed)  # for some custom operators
    np.random.seed(seed)  # for Numpy RNG
    cudnn.benchmark = True  # Disable for reproducability
    # cudnn.deterministic = True  # seed randomness used by cuDNN
    # torch.use_deterministic_algorithms(True)


def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:  # skip frozen weights
            continue
        if (
            len(param.shape) == 1
            or (name.endswith(".bias") and "stage" in name and "conv" in name)
            or name.endswith(".gain")  # skip biases in WS-Conv layers
            or name.endswith(".skipinit_gain")  # skip affine gains in WS-Conv layers
            or name in skip_list  # skip SkipInit gains
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def save_checkpoint(
    state, is_best, model, run_name, model_dir, filename="checkpoint", lastname=".pth"
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(os.path.join(model_dir, run_name)):
        os.makedirs(os.path.join(model_dir, run_name))

    epoch_num = state["epoch"]

    cp_path = os.path.join(model_dir, run_name, filename + lastname)

    torch.save(state, cp_path)

    if epoch_num % 1 == 0 and epoch_num >= 0:
        torch.save(state, cp_path.replace("checkpoint", "checkpoint_" + str(epoch_num)))

    if is_best:
        os.rename(
            cp_path.replace("checkpoint", "checkpoint_" + str(epoch_num)),
            cp_path.replace("checkpoint", "checkpoint_" + str(epoch_num) + "_best"),
        )
        print("New best model.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
