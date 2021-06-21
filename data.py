import os, random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np


class SingleImageDataset(Dataset):

    def __init__(self, img_dir, image_list_file, label_file, transform=None):
        self.imgs_first = load_imgs(img_dir, image_list_file, label_file)
        self.transform = transform

    def __getitem__(self, index):
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first

    def __len__(self):
        return len(self.imgs_first)


class SixImageDataset(Dataset):

    def __init__(self, img_dir, image_list_file, label_file, train, train_resize,
                 test_resize):
        (self.imgs_first, self.imgs_second, self.imgs_third, self.imgs_fourth,
         self.imgs_fifth, self.imgs_sixth) = load_imgs(img_dir,
                                                       image_list_file,
                                                       label_file,
                                                       num_images=6)
        # self.transform = transform
        self.train = train
        self.train_resize = train_resize
        self.test_resize = test_resize

    def transform(self, img_first, img_second, img_third, img_fourth, img_fifth,
                  img_sixth):
        if self.train:
            if random.random() > 0.5:
                img_first = TF.hflip(img_first)
            
            img_first = TF.resize(img_first, (self.train_resize[0] + 32, self.train_resize[1] + 32))
            angle = (random.random() - 0.5) * 2 * 30
            img_first = TF.rotate(img_first, angle)
            img_first = TF.center_crop(img_first, self.train_resize)
        else:
            img_first = TF.resize(img_first, self.test_resize)

        w, h = img_first.size
        img_second = TF.resized_crop(img_first, 0, 0, int(0.75 * h), int(0.75 * w),
                                     (w,h))
        img_third = TF.resized_crop(img_first, 0, int(0.25 * w), int(0.75 * h),
                                    int(0.75 * w), (w,h))
        img_fourth = TF.resized_crop(img_first, int(0.25 * h), int(0.125 * w),
                                     int(0.75 * h), int(0.75 * w), (w,h))
        img_fifth = TF.resized_crop(img_first, int(0.050 * h), int(0.050 * w),
                                    int(0.90 * h), int(0.90 * w), (w,h))
        img_sixth = TF.resized_crop(img_first, int(0.075 * h), int(0.075 * w),
                                    int(0.85 * h), int(0.85 * w), (w,h))

        input_norm_mean = [131.0912 / 255, 103.8827 / 255, 91.4953 / 255]
        input_norm_std = [1, 1, 1]
        img_first = TF.normalize(TF.to_tensor(img_first), input_norm_mean, input_norm_std)
        img_second = TF.normalize(TF.to_tensor(img_second), input_norm_mean,
                                  input_norm_std)
        img_third = TF.normalize(TF.to_tensor(img_third), input_norm_mean, input_norm_std)
        img_fourth = TF.normalize(TF.to_tensor(img_fourth), input_norm_mean,
                                  input_norm_std)
        img_fifth = TF.normalize(TF.to_tensor(img_fifth), input_norm_mean, input_norm_std)
        img_sixth = TF.normalize(TF.to_tensor(img_sixth), input_norm_mean, input_norm_std)

        return img_first, img_second, img_third, img_fourth, img_fifth, img_sixth

    def __getitem__(self, index):
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")

        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")

        path_fourth, target_fourth = self.imgs_fourth[index]
        img_fourth = Image.open(path_fourth).convert("RGB")

        path_fifth, target_fifth = self.imgs_fifth[index]
        img_fifth = Image.open(path_fifth).convert("RGB")

        path_sixth, target_sixth = self.imgs_sixth[index]
        img_sixth = Image.open(path_sixth).convert("RGB")

        img_first, img_second, img_third, img_fourth, img_fifth, img_sixth = \
            self.transform(img_first, img_second, img_third, img_fourth, img_fifth, img_sixth)

        return (img_first, target_first, img_second, target_second, img_third, target_third,
                img_fourth, target_fourth, img_fifth, target_fifth, img_sixth, target_sixth)

    def __len__(self):
        return len(self.imgs_first)


def load_imgs(img_dir, image_list_file, label_file, num_images=1):
    imgs_first = list()

    if num_images == 6:
        imgs_second = list()
        imgs_third = list()
        imgs_fourth = list()
        imgs_fifth = list()
        imgs_sixth = list()

    with open(image_list_file, "r") as imf:
        with open(label_file, "r") as laf:
            for line in imf:
                space_index = line.find(" ")
                video_name = line[0:space_index]
                video_path = os.path.join(img_dir, video_name)

                img_lists = os.listdir(video_path)
                record = laf.readline().strip().split()
                label = int(record[0])

                img_lists.sort()    # sort files by ascending
                img_path_first = video_path + "/" + img_lists[0]
                imgs_first.append((img_path_first, label))
                if num_images == 6:
                    img_path_second = video_path + "/" + img_lists[1]
                    img_path_third = video_path + "/" + img_lists[2]
                    img_path_fourth = video_path + "/" + img_lists[3]
                    img_path_fifth = video_path + "/" + img_lists[4]
                    img_path_sixth = video_path + "/" + img_lists[5]
                    imgs_second.append((img_path_second, label))
                    imgs_third.append((img_path_third, label))
                    imgs_fourth.append((img_path_fourth, label))
                    imgs_fifth.append((img_path_fifth, label))
                    imgs_sixth.append((img_path_sixth, label))

    if num_images == 1:
        return imgs_first
    elif num_images == 6:
        return imgs_first, imgs_second, imgs_third, imgs_fourth, imgs_fifth, imgs_sixth


def loader(img_dir,
           train_resize,
           test_resize,
           input_norm,
           num_inputs,
           batch_size,
           batch_size_t,
           workers,
           aug_option='v2'):
    train_img_dir = img_dir + "FER2013Train/"
    train_list_file = "txt/train_list.txt"
    train_label_file = "txt/train_label.txt"
    val_img_dir = img_dir + "FER2013Valid/"
    val_list_file = "txt/val_list.txt"
    val_label_file = "txt/val_label.txt"
    test_img_dir = img_dir + "FER2013Test/"
    test_list_file = "txt/test_list.txt"
    test_label_file = "txt/test_label.txt"

    if input_norm == 'imagenet':
        input_norm_mean = [0.485, 0.456, 0.406]
        input_norm_std = [0.229, 0.224, 0.225]
    elif input_norm == 'vggface2':
        input_norm_mean = [131.0912 / 255, 103.8827 / 255, 91.4953 / 255]
        input_norm_std = [1, 1, 1]

    if aug_option == 'v1':
        train_aug = transforms.Compose([
            transforms.Resize(test_resize),
            transforms.ToTensor(),
            transforms.Normalize(input_norm_mean, input_norm_std)
        ])
    elif aug_option == 'v2':
        train_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(train_resize),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(input_norm_mean, input_norm_std),
        ])
    elif aug_option == 'v3':
        train_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((train_resize[0] + 32, train_resize[1] + 32)),
            transforms.RandomRotation(30),
            transforms.CenterCrop(train_resize),
            transforms.ToTensor(),
            transforms.Normalize(input_norm_mean, input_norm_std),
        ])
        print('v3')

    test_aug = transforms.Compose([
        transforms.Resize(test_resize),
        transforms.ToTensor(),
        transforms.Normalize(input_norm_mean, input_norm_std)
    ])

    if num_inputs == 1:
        train_dataset = SingleImageDataset(train_img_dir, train_list_file, train_label_file, train_aug)
        val_dataset = SingleImageDataset(val_img_dir, val_list_file, val_label_file, test_aug)
        test_dataset = SingleImageDataset(test_img_dir, test_list_file, test_label_file, test_aug)
    elif num_inputs == 6:
        train_dataset = SixImageDataset(train_img_dir,
                                        train_list_file,
                                        train_label_file,
                                        train=True,
                                        train_resize=train_resize,
                                        test_resize=test_resize)
        val_dataset = SixImageDataset(val_img_dir,
                                      val_list_file,
                                      val_label_file,
                                      train=False,
                                      train_resize=train_resize,
                                      test_resize=test_resize)
        test_dataset = SixImageDataset(test_img_dir,
                                       test_list_file,
                                       test_label_file,
                                       train=False,
                                       train_resize=train_resize,
                                       test_resize=test_resize)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_t,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_t,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader#, train_dataset, test_dataset


if __name__ == '__main__':
    img_dir = 'C:/Users/VTRG/Desktop/ferplus_detected/data/'
    # img_dir = 'C:/Users/nSamsow/ferplus_detected/data/'
    
    input_norm_mean = [131.0912 / 255, 103.8827 / 255, 91.4953 / 255]
    input_norm_std = [1, 1, 1]

    _, _, _, train_dataset, test_dataset = loader(
        img_dir, (224, 224), (224, 224), 'vggface2', 
        6, 32, 32, 8, aug_option='v2')

    # for i in range(40):
    #     img = train_dataset.__getitem__(index=i)[0]
    #     img = img.permute(1, 2, 0).numpy()
    #     img = (img * np.array(input_norm_std) + np.array(input_norm_mean)) * 255
    #     img = img.astype(np.uint8)
    #     plt.subplot(4, 10, i+1)
    #     plt.imshow(img)
    for i in range(0, 6, 2):
        img = train_dataset.__getitem__(index=i)
        for j in range(6):
            im = img[j * 2]
            im = im.permute(1, 2, 0).numpy()
            im = (im * np.array(input_norm_std) + np.array(input_norm_mean)) * 255
            im = im.astype(np.uint8)
            plt.subplot(6, 6, i * 6 + j + 1)
            plt.imshow(im)

        img = test_dataset.__getitem__(index=i)
        for j in range(6):
            im = img[j * 2]
            im = im.permute(1, 2, 0).numpy()
            im = (im * np.array(input_norm_std) + np.array(input_norm_mean)) * 255
            im = im.astype(np.uint8)
            plt.subplot(6, 6, (i + 1) * 6 + j + 1)
            plt.imshow(im)

    plt.show()
