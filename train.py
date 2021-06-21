import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from timm.models import model_parameters
from timm.utils import dispatch_clip_grad
from wandb import AlertLevel

from data import loader
from models.models import *
from models.rbloss import RBLoss
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model_names = [
    "resnet18",
    "resnet50",
    "nfnet_l0",
    "ran_resnet18",
    "ran_resnet50",
    "ran_nfnet_l0",
    "my_resnet18",
    "my_resnet50",
    "my_resnet50_ox",
    "my_nfnet_l0",
]


parser = argparse.ArgumentParser(description="Napat's Senior Project")
parser.add_argument(
    "--img_dir",
    metavar="DIR",
    default="C:/Users/GPU/Desktop/ferplus_detected/data/",
    type=str,
    help="path to dataset",
)  # C:/Users/nSamsow/ferplus_detected/data/
parser.add_argument("-m", "--model_dir", default="model_cp/", type=str)
parser.add_argument(
    "-j",
    "--workers",
    metavar="N",
    default=8,
    type=int,
    help="number of data loading workers (default: 4)",
)

parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="nfnet_l0",
    type=str,
    choices=model_names,
    help="model architecture",
)
parser.add_argument(
    "-b",
    "--batch-size",
    metavar="N",
    default=32,
    type=int,
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "-b_t",
    "--batch-size_t",
    metavar="N",
    default=32,
    type=int,
    help="mini-batch size (default: 256)",
)
parser.add_argument("--input-norm", default="vggface2", type=str)
parser.add_argument("--aug", default="v3", type=str)

parser.add_argument("--momentum", metavar="M", default=0.9, type=float, help="momentum")
parser.add_argument(
    "--wd", "--weight-decay", metavar="W", default=1e-5, type=float, help="weight decay"
)
parser.add_argument(
    "--clip-value",
    metavar="CV",
    default=0.0,
    type=float,
    help="gradient clipping value",
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    metavar="LR",
    default=0.01,
    type=float,
    help="initial learning rate",
)
parser.add_argument(
    "--epochs", metavar="N", default=40, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    metavar="N",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--milestones", metavar="MS", default="15,30", type=str, help="milestones"
)
parser.add_argument(
    "--lr-decay",
    metavar="LRD",
    default=0.1,
    type=float,
    help="learning rate decay ratio",
)
parser.add_argument("--warmup-t", metavar="N", default=0, type=int)
parser.add_argument("--t-initial", metavar="N", default=0, type=int)

parser.add_argument(
    "--model-ema",
    action="store_true",
    default=False,
    help="Enable tracking moving average of model weights",
)
parser.add_argument(
    "--model-ema-force-cpu",
    action="store_true",
    default=False,
    help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
)
parser.add_argument(
    "--model-ema-decay",
    type=float,
    default=0.99999,
    help="decay factor for model weights moving average (default: 0.9998)",
)

parser.add_argument(
    "--checkpoint",
    metavar="PATH",
    default="",
    type=str,
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    metavar="PATH",
    default="",
    type=str,
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--pretrained",
    metavar="PATH",
    default="",
    type=str,
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)


set_seed(seed=555)
BEST_PREC1 = 0
BEST_PREC1_EMA = 0


def main():
    global BEST_PREC1, BEST_PREC1_EMA, args, run_name
    args = parser.parse_args()

    train_resize = (224, 224)
    test_resize = (224, 224) if "l0" not in args.arch else (288, 288)

    num_inputs = 6 if any(x in args.arch for x in ["ran", "my"]) else 1
    print("#Inputs:", num_inputs)

    train_loader, val_loader, test_loader = loader(
        args.img_dir,
        train_resize,
        test_resize,
        args.input_norm,
        num_inputs,
        args.batch_size,
        args.batch_size_t,
        args.workers,
        aug_option=args.aug,
    )

    ## Prepare model
    assert args.arch in model_names
    if args.arch == "resnet18":
        model = resnet18()
    elif args.arch == "resnet50":
        model = resnet50()
    elif args.arch == "nfnet_l0":
        model = nfnet_l0()
    elif args.arch == "ran_resnet18":
        model = ran_resnet18()
    elif args.arch == "ran_resnet50":
        model = ran_resnet50()
    elif args.arch == "ran_nfnet_l0":
        model = ran_nfnet_l0()
    elif args.arch == "my_resnet18":
        model = my_resnet18()
    elif args.arch == "my_resnet50":
        model = my_resnet50()
    elif args.arch == "my_resnet50_ox":
        model = my_resnet50_ox()
    elif args.arch == "my_nfnet_l0":
        model = my_nfnet_l0()

    if args.checkpoint:
        cp = torch.load(args.checkpoint)
        model.load_state_dict(cp["state_dict"], strict=True)
        print("Load checkpoint completed!")

    model.cuda()

    ## Define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_rb = RBLoss().cuda()

    parameters = model.parameters()
    weight_decay = args.wd
    # if 'l0' in args.arch or 'ox' in args.arch:
    if weight_decay:
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.0

    optimizer = torch.optim.SGD(
        parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay
    )
    warmup_t = args.warmup_t
    milestones = [int(item) for item in args.milestones.split(",")]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_decay, verbose=True
    )

    model_ema = None

    if args.pretrained:
        checkpoint = torch.load(
            # "/home/kwang/AAAI/Emotion18_task1/Data/Model/ijba_res18_naive.pth.tar"
            "ijba_res18_naive.pth.tar"
        )

        pretrained_state_dict = checkpoint["state_dict"]
        model_state_dict = model.state_dict()

        for key in pretrained_state_dict:
            if (key == "module.fc.weight") | (key == "module.fc.bias"):
                pass
                # if  ((key=='module.fc.weight')|(key=='module.fc.bias')|(key == 'module.feature.weight')|(key == 'module.feature.bias')):
            else:
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict, strict=False)

    ## Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            BEST_PREC1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ## Only evaluate the model
    if args.evaluate:
        print(f"Model: {args.arch}\nCheckpoint: {args.checkpoint}")
        validate(
            test_loader,
            model,
            criterion_ce,
            criterion_rb,
            mode="Test",
            log_suffix=" (Test)",
        )
        return

    wandb.init(
        project="fer-senior",
        config={
            # Dataset / Model parameters
            "ARCH": args.arch,
            "Batch Size": args.batch_size,
            "Input Normalization": args.input_norm,
            "Data Augmentation": args.aug,
            # Optimizer parameters
            "Optimizer": optimizer,
            "Momentum": args.momentum,
            "Weight Decay": args.wd,
            "Clip Value": args.clip_value,
            # Learning rate schedule parameters
            "Scheduler": scheduler,
            "LR": args.lr,
            "#Epochs": args.epochs,
            "Start Epoch": args.start_epoch,
            "Milestones": args.milestones,
            "LR Decay Factor": args.lr_decay,
            "#Warmup Epochs": args.warmup_t,
            "Init. Cosine Period": args.t_initial,
            # Model Exponential Moving Average
            "Model EMA": args.model_ema,
            "Model EMA Max. Decay": args.model_ema_decay,
            # Others
            "Resume": args.resume,
            "Pretrained": args.pretrained,
            "Evaluate": args.evaluate,
        },
    )
    wandb.watch(model, log="all", log_freq=2000)

    ## Optimization loop
    for epoch in range(args.start_epoch, args.epochs):  # 0 to #epochs - 1
        ## Train for one epoch
        train(
            epoch,
            args.epochs,
            model,
            train_loader,
            optimizer,
            criterion_ce,
            criterion_rb,
            args,
            scheduler,
            model_ema,
            warmup_t,
        )

        ## Evaluate on val. set check if it's better and save checkpoint
        prec1 = validate(
            val_loader,
            model,
            criterion_ce,
            criterion_rb,
            mode="Validation",
            log_suffix="",
        )
        is_best = False
        if prec1 >= BEST_PREC1:
            BEST_EPOCH = epoch + 1
            BEST_PREC1 = prec1
            is_best = True

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": BEST_PREC1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            model,
            run_name,
            model_dir=args.model_dir,
        )

        scheduler.step()

    print(f"Test: BEST EPOCH = {BEST_EPOCH}")
    cp_folder = os.path.join(args.model_dir, run_name)
    state_dict = torch.load(
        os.path.join(cp_folder, f"checkpoint_{BEST_EPOCH}_best.pth")
    )["state_dict"]
    model.load_state_dict(state_dict)
    validate(test_loader, model, criterion_ce, criterion_rb, mode="Test", log_suffix="")
    if model_ema is not None:
        validate(
            test_loader,
            model_ema.module,
            criterion_ce,
            criterion_rb,
            mode="Test",
            log_suffix=" (EMA)",
        )


def train(
    epoch,
    epochs,
    model,
    train_loader,
    optimizer,
    criterion_ce,
    criterion_rb,
    args,
    scheduler,
    model_ema,
    warmup_t,
):
    print_time = AverageMeter()
    training_time = AverageMeter()
    ce_losses = AverageMeter()
    rb_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    start_training_time = time.time()
    iters = len(train_loader)
    for (i, pack) in enumerate(train_loader):
        if ("ran" not in args.arch) and ("my" not in args.arch):
            input_first, target_first = pack
            input_ = input_first
        else:
            (
                input_first,
                target_first,
                input_second,
                _,
                input_third,
                _,
                input_forth,
                _,
                input_fifth,
                _,
                input_sixth,
                _,
            ) = pack
            input_ = torch.zeros(
                [
                    input_first.shape[0],
                    input_first.shape[1],
                    input_first.shape[2],
                    input_first.shape[3],
                    6,
                ]
            )
            input_[:, :, :, :, 0] = input_first
            input_[:, :, :, :, 1] = input_second
            input_[:, :, :, :, 2] = input_third
            input_[:, :, :, :, 3] = input_forth
            input_[:, :, :, :, 4] = input_fifth
            input_[:, :, :, :, 5] = input_sixth

        ## Store input_ and target in CUDA memory
        input_ = input_.cuda(non_blocking=True)
        target = target_first.cuda(non_blocking=True)

        ## Automatic differentiation
        # input_var = torch.autograd.Variable(input_)
        # target_var = torch.autograd.Variable(target)

        ## Compute output
        output = model(input_)

        ## Calculate losses
        if ("ran" not in args.arch) and ("my" not in args.arch):
            pred_score = output
            ce_loss = criterion_ce(pred_score, target)
            rb_loss = torch.Tensor([0]).cuda()
            loss = ce_loss
        else:
            pred_score, alphas_part_max, alphas_org = output
            ce_loss = criterion_ce(pred_score, target)
            rb_loss = criterion_rb(alphas_part_max, alphas_org)
            loss = ce_loss + rb_loss

        ## Measure accuracy
        prec1 = accuracy(pred_score.data, target, topk=(1,))

        ## Update loss and accuracy
        ce_losses.update(ce_loss.item(), n=input_.size(0))
        rb_losses.update(rb_loss.item(), n=input_.size(0))
        losses.update(loss.item(), n=input_.size(0))
        top1.update(prec1[0], n=input_.size(0))

        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if args.clip_value != 0:
            dispatch_clip_grad(
                model_parameters(model, exclude_head=True),
                value=args.clip_value,
                mode="agc",
            )
        optimizer.step()

        dk = None
        if model_ema is not None:
            dk = min(
                args.model_ema_decay,
                (1 + (epoch * iters + i) / args.batch_size)
                / (10 + (epoch * iters + i) / args.batch_size),
            )
            model_ema.decay = 1 - dk
            model_ema.update(model)

        # scheduler.step(epoch + i / iters)

        ## Training log
        wandb.log(
            {
                "Training CE Loss Now": ce_losses.val,
                "Training RB Loss Now": rb_losses.val,
                "Training Loss Now": losses.val,
                "Training Accuracy Now": top1.val,
                "Learning Rate": optimizer.param_groups[0]["lr"],
                "EMA Decay": dk,
            }
        )

        ## Pretty print
        if i % int(len(train_loader) / 20) == 0:
            print_time.update(time.time() - end)
            end = time.time()
            print(
                "[% 2d/%d][% 3d/%d]\tCE Loss: %.5f,\tRB Loss: %.5f,\tLoss: %.5f,\tTime: % 5.1f (% 5.1f) sec"
                % (
                    epoch + 1,
                    epochs,
                    i,
                    len(train_loader),
                    ce_losses.val,
                    rb_losses.val,
                    losses.val,
                    print_time.val,
                    print_time.avg,
                )
            )

    training_time.update(time.time() - start_training_time)

    wandb.log(
        {
            "Current Epoch": epoch,
            "Training Accuracy": top1.avg,
            "Avg. Training CE Loss": ce_losses.avg,
            "Avg. Training RB Loss": rb_losses.avg,
            "Avg. Training Loss": losses.avg,
        }
    )

    print("Training: ")
    print(
        "\tAccuracy: % 5.2f,\tAvg. Loss: %.5f,\tAvg. CE Loss: %.5f,\tAvg. RB Loss: %.5f\tTime: % 5.1f sec"
        % (top1.avg, losses.avg, ce_losses.avg, rb_losses.avg, training_time.val)
    )


def validate(data_loader, model, criterion_ce, criterion_rb, mode, log_suffix):
    val_time = AverageMeter()
    ce_losses = AverageMeter()
    rb_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for (i, pack) in enumerate(data_loader):
            if ("ran" not in args.arch) and ("my" not in args.arch):
                input_first, target_first = pack
                input_ = input_first
            else:
                (
                    input_first,
                    target_first,
                    input_second,
                    _,
                    input_third,
                    _,
                    input_forth,
                    _,
                    input_fifth,
                    _,
                    input_sixth,
                    _,
                ) = pack
                input_ = torch.zeros(
                    [
                        input_first.shape[0],
                        input_first.shape[1],
                        input_first.shape[2],
                        input_first.shape[3],
                        6,
                    ]
                )
                input_[:, :, :, :, 0] = input_first
                input_[:, :, :, :, 1] = input_second
                input_[:, :, :, :, 2] = input_third
                input_[:, :, :, :, 3] = input_forth
                input_[:, :, :, :, 4] = input_fifth
                input_[:, :, :, :, 5] = input_sixth

            ## Store input_ and target in CUDA memory
            input_ = input_.cuda(non_blocking=True)
            target = target_first.cuda(non_blocking=True)

            ## Automatic differentiation
            # input_var = torch.autograd.Variable(input_)
            # target_var = torch.autograd.Variable(target)

            ## Compute output
            output = model(input_)

            ## Calculate losses
            if ("ran" not in args.arch) and ("my" not in args.arch):
                pred_score = output
                ce_loss = criterion_ce(pred_score, target)
                rb_loss = torch.Tensor([0]).cuda()
                loss = ce_loss
            else:
                pred_score, alphas_part_max, alphas_org = output
                ce_loss = criterion_ce(pred_score, target)
                rb_loss = criterion_rb(alphas_part_max, alphas_org)
                loss = ce_loss + rb_loss

            ## Measure accuracy
            prec1 = accuracy(pred_score.data, target, topk=(1,))

            ## Update loss and accuracy
            ce_losses.update(ce_loss.item(), n=input_.size(0))
            rb_losses.update(rb_loss.item(), n=input_.size(0))
            losses.update(loss.item(), n=input_.size(0))
            top1.update(prec1[0], n=input_.size(0))

    val_time.update(time.time() - end)

    ## Validation log
    if not args.evaluate:
        wandb.log(
            {
                f"{mode} Accuracy{log_suffix}": top1.avg,
                f"Avg. {mode} CE Loss{log_suffix}": ce_losses.avg,
                f"Avg. {mode} RB Loss{log_suffix}": rb_losses.avg,
                f"Avg. {mode} Loss{log_suffix}": losses.avg,
            }
        )

    print(f"{mode}:")
    print(
        "\tAccuracy: % 5.2f,\tAvg. Loss: %.5f,\tAvg. CE Loss: %.5f,\tAvg. RB Loss: %.5f\tTime: % 5.1f sec"
        % (top1.avg, losses.avg, ce_losses.avg, rb_losses.avg, val_time.val)
    )

    return top1.avg


if __name__ == "__main__":
    global args, start_time, run_name

    start_time = datetime.now()

    args = parser.parse_args()

    run_name = "%s-arch-%s-b-%d-lr-%.2e-clip-%.2f-epochs-%d-wd-%f" % (
        start_time.strftime("%Y%m%d_%H%M%S"),
        args.arch,
        args.batch_size,
        args.lr,
        args.clip_value,
        args.epochs,
        args.wd,
    )

    print("Using {} device".format(device))
    main()
    print(".\nDone!")
    print_total_time(datetime.now() - start_time)

    if not args.checkpoint:
        print("-" * 160, "\n", "-" * 160)

        wandb.alert(
            title="Finished!",
            text=f"run-{run_name} has been FINISHED!",
            level=AlertLevel.INFO,
        )
        wandb.finish()
