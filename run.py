import subprocess
import time



bs = 8
wd = 1e-6
clip_value = 0
lr = 3e-3
t = 60
subprocess.run([
"cmd", "/c", f"python trainnf.py -j 4 "
f"-a my_resnet50_ox --clip-value {clip_value} "
f"-b {bs} -b_t {int(bs / 2)} --input-norm vggface2 --aug v3 "
f"--lr {lr} --epochs {t} --milestones 15,30,45 --lr-decay 0.1 --momentum 0.9 --wd {wd} "])


# bs = 8
# wd = 1e-6
# clip_value = 0
# lr = 3e-3
# t = 40
# warmup_t = t_initial = 0
# subprocess.run([
# "cmd", "/c", f"python trainnf.py -j 4 "
# f"-a ran_resnet50 --clip-value {clip_value} "
# f"-b {bs} -b_t {int(bs / 2)} --input-norm vggface2 --aug v3 "
# f"--lr {lr} --epochs {t} --milestones 15,30 --lr-decay 0.1 --momentum 0.9 --wd {wd} "])


# bs = 8
# wd = 1e-6
# clip_value = 0.48
# lr = 3e-3
# t = 40
# warmup_t = t_initial = 0
# subprocess.run([
# "cmd", "/c", f"python trainnf.py -j 4 "
# f"-a my_nfnet_l0 --clip-value {clip_value} "
# f"-b {bs} -b_t {int(bs / 2)} --input-norm vggface2 --aug v3 "
# f"--lr {lr} --epochs {t} --milestones 15,30 --lr-decay 0.1 --momentum 0.9 --wd {wd} "])


# bs = 8
# wd = 1e-5
# clip_value = 0
# lr = 1e-4
# t = 25
# warmup_t = t_initial = 0
# subprocess.run([
# "cmd", "/c", f"python trainnf.py " 
# f"-a ran_resnet50 -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
# f"--momentum 0.9 --wd {wd} --clip-value {clip_value} "
# f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])

# bs = 8
# wd = 1e-5
# clip_value = 0.64
# lr = 1e-3
# t = 40
# warmup_t = t_initial = 0
# subprocess.run([
# "cmd", "/c", f"python trainnf.py " 
# f"-a nfnet_l0 -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
# f"--momentum 0.9 --wd {wd} --clip-value {clip_value} "
# f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])


# bs = 8
# wd = 5e-5
# # clip_value = 0.64
# lr = 1e-3
# t = 50
# warmup_t = t_initial = 0
# # for model in ['my_nfnet_l0']:
# subprocess.run([
# "cmd", "/c", f"python trainnf.py " 
# f"-a my_resnet18 -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
# f"--momentum 0.9 --wd {wd} --clip-value 0 "
# f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])
# subprocess.run([
# "cmd", "/c", f"python trainnf.py " 
# f"-a my_resnet50 -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
# f"--momentum 0.9 --wd {wd} --clip-value 0 "
# f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])
# subprocess.run([
# "cmd", "/c", f"python trainnf.py " 
# f"-a my_nfnet_l0 -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
# f"--momentum 0.9 --wd {wd} --clip-value 0.64 "
# f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])


## RAN
# NFNet-L0
# bs = 8
# wd = 5e-4
# clip_value = 0.48
# lr = 1e-2
# t = 45
# warmup_t = t_initial = 0
# for model in ['ran_nfnet_l0']:
#     subprocess.run([
#     "cmd", "/c", f"python trainnf.py " 
#     f"-a {model} -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
#     f"--momentum 0.9 --wd {wd} --clip-value {clip_value} "
#     f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])


## Pure Architecture: 
# # ResNet
# bs = 32
# wd = 5e-4
# clip_value = 0
# lr = 1e-2
# t = 45
# warmup_t = t_initial = 0
# for model in ['resnet18', 'resnet50']:
#     subprocess.run([
#     "cmd", "/c", f"python trainnf.py " 
#     f"-a {model} -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
#     f"--momentum 0.9 --wd {wd} --clip-value {clip_value} "
#     f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])

# # NFNet-L0: round 4
# bs = 32
# wd = 5e-4
# clip_value = 0.48
# lr = 3e-3
# t = 45
# warmup_t = t_initial = 0
# subprocess.run([
# "cmd", "/c", f"python trainnf.py " 
# f"-a nfnet_l0 -b {bs} -b_t {bs} --input-norm vggface2 --aug v2 "
# f"--momentum 0.9 --wd {wd} --clip-value {clip_value} "
# f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "])


# # NFNet-L0: round 3
# bs = 32
# wd = 3e-4
# clip_value = 0.48
# lrs  = [8e-4, 7e-4, 6e-4, 5e-4]   # 1e-3, 9e-4, 6e-4, 4e-4, 2.5e-4, 1.6e-4
# warmup_t = 5
# t = t_initial = 40
# for lr in lrs:
#     subprocess.run([
#         "cmd", "/c", f"python trainnf.py " 
#         f"-a nfnet_l0 -b {bs} -b_t {bs} --input-norm imagenet --aug v2 "
#         f"--momentum 0.9 --wd {wd} --clip-value {clip_value} "
#         f"--lr {lr} --epochs {t} --warmup-t {warmup_t} --t-initial {t_initial} "
#         f"--model-ema --model-ema-decay 0.99999 "])

# NFNet-L0: round 2
# bs = 32
# # lr = 2e-2
# wd = 3e-4
# t = 10
# t_initial = 60
# # subprocess.run(["cmd", "/c", 
# # f"python trainnf.py -a nfnet_l0 --lr 1e-3 -b {bs} -b_t {bs} --wd {wd} --epochs {t} --t-initial {t_initial} --warmup-t 5 --clip-value 0.48"])
# subprocess.run(["cmd", "/c", 
# f"python trainnf.py -a nfnet_l0 --lr 1e-4 -b {bs} -b_t {bs} --wd {wd} --epochs {t} --t-initial {t_initial} --warmup-t 5 --clip-value 0.48"])
# subprocess.run(["cmd", "/c", 
# f"python trainnf.py -a nfnet_l0 --lr 1e-3 -b {bs} -b_t {bs} --wd {wd} --epochs {t} --t-initial {t_initial} --warmup-t 2 --clip-value 0.48"])
# subprocess.run(["cmd", "/c", 
# f"python trainnf.py -a nfnet_l0 --lr 1e-4 -b {bs} -b_t {bs} --wd {wd} --epochs {t} --t-initial {t_initial} --warmup-t 2 --clip-value 0.48"])
# for warmup_t in [5, 2]:
#     # for lr in [0.8e-2, 1e-2, 1.25e-2, 1.6e-2]:
#     for lr in [1e-3, 1e-4]:
#         subprocess.run(["cmd", "/c", 
        # f"python trainnf.py -a nfnet_l0 --lr {lr} -b {bs} -b_t {bs} --wd {wd} --epochs {t} --t-initial {t_initial} --warmup-t {warmup_t} --clip-value 0.48"])

# NFNet-L0: round 1
# bs = 32
# lr = 1e-2
# wd = 3e-4
# t = 20
# t_initial = 60
# subprocess.run(["cmd", "/c", 
# f"python trainnf.py -a nfnet_l0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={t} --t-initial={t_initial} --clip-value=0.48 --model-ema --model-ema-decay=0.99999"])
# subprocess.run(["cmd", "/c", 
# f"python trainnf.py -a nfnet_l0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={t} --t-initial={t_initial} --clip-value=0.32 --model-ema --model-ema-decay=0.99999"])
# subprocess.run(["cmd", "/c", 
# f"python trainnf.py -a nfnet_l0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={t} --t-initial={t_initial} --clip-value=0.40 --model-ema --model-ema-decay=0.99999"])

# NFNet-F0
# subprocess.run(["cmd", "/c", f"python trainnf.py -a dm_nfnet_f0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs} --clip-value=0.64"])
# subprocess.run(["cmd", "/c", f"python trainnf.py -a dm_nfnet_f0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs} --clip-value=0.48"])
# subprocess.run(["cmd", "/c", f"python trainnf.py -a dm_nfnet_f0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs} --clip-value=0.32"])


## Fine tune ResNet50: Pause
# bs = 32
# lr = 1e-4
# wd = 1e-3
# epochs = 60
# subprocess.run(["cmd", "/c", f"python train.py -a resnet50_ft --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# bs = 32
# # lr = 1e-2
# # wd = 3e-4
# epochs = 5
# for lr in [1e-5, 1e-4, 1e-3]:
#     for wd in [1e-5, 1e-4]:
#         subprocess.run(["cmd", "/c", f"python train.py -a resnet50_ft --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])

# ## Pure Architecture: 2
# bs = 32
# lr = 1e-2
# wd = 3e-4
# epochs = 60
# subprocess.run(["cmd", "/c", f"python train.py -a resnet18 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a resnet34 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a resnet50 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a seresnext50 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])


# ## Pure Architecture
# lr = 1e-2
# bs = 32
# wd = 1e-4
# epochs = 50
# subprocess.run(["cmd", "/c", f"python train.py -a resnet18 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a resnet34 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a resnet50 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a seresnext50 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])


# ## Pure Architecture
# lr = 1e-2
# bs = 32
# wd = 2e-5
# epochs = 75
# subprocess.run(["cmd", "/c", f"python train.py -a dm_nfnet_f0 --clip-value=0.32 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a dm_nfnet_f0 --clip-value=0.16 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a seresnext50 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a resnet50 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a resnet34 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])
# subprocess.run(["cmd", "/c", f"python train.py -a resnet18 --clip-value=0 --lr={lr} -b {bs} -b_t {bs} --wd={wd} --epochs={epochs}"])


## Pure Architecture: Tune
# for cv in [0.04, 0.08, 0.16, 0.32, 0.48, 0.64, 0.96, 1.28]:
#     subprocess.run(["cmd", "/c", f"python train.py -a dm_nfnet_f0 --clip-value={cv} --lr=1e-2 -b 32 -b_t 32 --wd=2e-5 --epochs=8"])
# for cv in [0.32, 0.40, 0.48, 0.64, 0.96, 1.28]:
#     subprocess.run(["cmd", "/c", f"python train.py -a dm_nfnet_f0 --clip-value={cv} --lr=1e-2 -b 32 -b_t 32 --wd=1e-6 --epochs=5"])
# for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
#     subprocess.run(["cmd", "/c", f"python train.py -a dm_nfnet_f0 --lr={lr} -b 32 -b_t 32 --wd=1e-6 --epochs=5"])
# for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
#     subprocess.run(["cmd", "/c", f"python train.py -a seresnext50 --lr={lr} -b 32 -b_t 32 --wd=1e-6 --epochs=5"])
# python train.py -a dm_nfnet_f0 --clip-value=0.32 --lr=1e-4 -b 64 -b_t 64 --wd=1e-5 --epochs=40
# python train.py -a dm_nfnet_f0 --clip-value=0.32 --lr=1e-4 -b 32 -b_t 32 --wd=1e-4 --epochs=40
# python train.py -a dm_nfnet_f0 --clip-value=0.32 --lr=1e-3 -b 32 -b_t 32 --wd=1e-4 --epochs=40


## Pure Architecture
# epochs = 40
# lrs = [1e-5, 3e-6, 1e-6]
# model_names = ['dm_nfnet_f0', 'seresnext50', 'resnet50', 'resnet34', 'resnet18']
# for model in model_names:
#     for lr in lrs:
#         count_down()
#         subprocess.run(["cmd", "/c", f"python train.py -a {model} --epochs={epochs} -b 8 -b_t 8 --lr={lr}"])


## Pure Architecture: Check LR
# epochs = 5
# lrs = [3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
# model_names = ['dm_nfnet_f0', 'seresnext50', 'resnet50', 'resnet34', 'resnet18']
# for model in model_names:
#     for lr in lrs:
#         count_down()
#         subprocess.run(["cmd", "/c", f"python train.py -a {model} --epochs={epochs} -b 8 -b_t 8 --lr={lr}"])


# epochs = 40
# lr = [1e-4, 1e-5, 1e-6]
# # Check
# count_down()
# subprocess.run(["cmd", "/c", f"python train.py -a ran_dm_nfnet_f0 --epochs={epochs} -b 8 -b_t 4 --lr={lr[0]}"])
# count_down()
# subprocess.run(["cmd", "/c", f"python train.py -a ran_seresnext50 --epochs={epochs} -b 8 -b_t 4 --lr={lr[0]}"])
# count_down()
# subprocess.run(["cmd", "/c", f"python train.py -a ran_resnet50 --epochs={epochs} -b 8 -b_t 4 --lr={lr[0]}"])
