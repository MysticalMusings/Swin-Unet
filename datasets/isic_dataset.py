import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd


class IsicDataset(Dataset):
    def __init__(
        self,
        path="/root/autodl-tmp/maiden/data/isic",
        img_size=224,
        mode="train",
        tran_list=None,
        augment=False,
        *args,
        **kwargs,
    ):
        mode_path = "Training" if mode != "test" else "Test"
        df = pd.read_csv(
            os.path.join(
                path,
                f"ISBI2016_ISIC_Part3B_{mode_path}_GroundTruth.csv",
            )
        )
        self.name_list = df.iloc[:, 0].tolist()
        self.label_list = df.iloc[:, 1].tolist()
        self.path = os.path.join(path, f"ISBI2016_ISIC_Part3B_{mode_path}_Data")
        self.mode = mode
        self.img_size = img_size
        
        if tran_list is None:
            tran_list = [transforms.ToTensor()]
        if augment:
            tran_list += [
                # transforms.RandomRotation(180, expand=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        self.tran_list = tran_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.path, name + ".jpg")

        msk_path = os.path.join(self.path, name + "_Segmentation.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        state = torch.get_rng_state()
        img = trans_norm(
            img,
            self.tran_list + [transforms.Resize((self.img_size, self.img_size))],
            state,
            norm=False,
            rescale=False,
        )
        mask = trans_norm(
            mask,
            self.tran_list
            + [
                transforms.Resize(
                    (self.img_size, self.img_size),
                    interpolation=(
                        transforms.InterpolationMode.NEAREST
                    ),
                )
            ],
            state,
            norm=False,
            rescale=False,
        )
        # img = torch.concat((img, mask))

        return {
            "image": img,
            "index": index,
            "label": mask[0],
            "case_name": img_path.split('/')[-1],
            "mask_path": msk_path,
        }

def trans_norm(tensor, tran_list=None, state=None, norm=True, rescale=True):
    """apply transformation and normalization to tensors into range (-1,1)"""
    if tran_list:
        if state is not None:
            torch.set_rng_state(state)
        tensor = transforms.Compose(tran_list)(tensor)
    if rescale and tensor.unique().numel() > 1:
        other_axes = [i for i in range(1, len(tensor.shape))]
        _min = tensor.amin(dim=other_axes, keepdim=True)
        _max = tensor.amax(dim=other_axes, keepdim=True)
        tensor = (tensor - _min) / (_max - _min + 1e-6)  # range (0, 1)
    if norm:
        normalizer = transforms.Normalize(
            tuple([0.5] * tensor.shape[0]), tuple([0.5] * tensor.shape[0])
        )
        tensor = normalizer(tensor)  # range (-1, 1)
    return tensor
