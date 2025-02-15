import monai as mn
import glob
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
from random import shuffle, seed
import custom_cc
import custom

seed(786)


def add_bg(x):
    # print('DEBUG INSIDE MONAI TRANSFORM')
    # print(x.shape)
    if len(x.shape) == 3:
        if x.size(0) == 1:
            x = x[..., None]
        else:
            x = x[None]
    return x


def get_loaders(
    batch_size=1,
    device="cpu",
    lowres=False,
    num_ch=1,
    local_paths=False,
    mni=False,
):
    if os.path.exists("./data_list.txt"):
        my_file = open("./data_list.txt", "r")
        data = my_file.read()
        train_files = data.split("\n")
        my_file.close()
    elif local_paths:
        pdir = "/run/user/2061/gvfs/smb-share:server=isis,share=language_ashburner/Clinical_scans_Temp/"
        train_files = glob.glob(os.path.join(pdir, "*/MRI_Nifti/*.json"))
        # +\
        #                 glob.glob(os.path.join(pdir, '*/MRI_Nifti/*/*.json')) +\
        #                 glob.glob(os.path.join(pdir, '*/*/MRI_Nifti/*.json')) +\
        #                 glob.glob(os.path.join(pdir, '*/*/MRI_Nifti/*/*.json')) +\
        #                 glob.glob(os.path.join(pdir, '*/MRI_nifti/*.json')) +\
        #                 glob.glob(os.path.join(pdir, '*/MRI_nifti/*/*.json')) +\
        #                 glob.glob(os.path.join(pdir, '*/*/MRI_nifti/*.json')) +\
        #                 glob.glob(os.path.join(pdir, '*/*/MRI_nifti/*/*.json'))
    else:
        raise NotImplementedError(
            "'Baseline' uses private stroke data and so requires --local_paths ."
        )

    grouped_files = []
    files = []
    base = ""
    for i, f in enumerate(train_files):
        base_ = "/".join(f.split("/")[:-1])
        if base_ != base or i == len(train_files):
            if len(files) > 0:
                grouped_files.append(files)
            base = base_
            files = []
        files.append(f)

    train_dict_ = [
        {
            "image": [
                f.replace(".json", ".nii").replace("ifti/", "ifti/aa_") for f in g
            ],
            "params": [f for f in g],
        }
        for g in grouped_files
    ]

    train_dict = []
    for item in train_dict_:
        cnt = 0
        img_ = []
        meta_ = []
        for img, meta in zip(item["image"], item["params"]):
            if os.path.exists(img):
                cnt += 1
                img_.append(img)
                meta_.append(meta)
            # else:
            #     print(img)
        if cnt > 0:
            train_dict.append({"image": img_, "params": meta_})

    shuffle(train_dict)

    train_dict, val_dict = (
        train_dict[:-100],
        train_dict[-100:],
    )

    ptch = 96 if lowres else 192

    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.CopyItemsD(keys=["image"], names=["path"]),
            mn.transforms.LoadImageD(keys=["image"], image_only=False),
            custom_cc.LoadMetadataD(keys=["params"]),
            mn.transforms.EnsureChannelFirstD(keys=["image"]),
            mn.transforms.ForegroundMaskD(
                keys=["image"], invert=True, new_key_prefix="mask_"
            ),
            mn.transforms.OrientationD(keys=["image", "mask_image"], axcodes="RAS"),
            mn.transforms.CopyItemsD(keys=["image"], names=["target"]),
            mn.transforms.SpacingD(
                keys=["image", "mask_image"], pixdim=2 if lowres else 1
            ),
            mn.transforms.ToTensorD(
                dtype=float, keys=["image", "mask_image", "params", "target"]
            ),
            mn.transforms.RandSpatialCropD(
                keys=["image", "mask_image"], roi_size=ptch, random_size=False
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["image", "mask_image"], spatial_size=(ptch, ptch, ptch)
            ),
            # mn.transforms.RandAxisFlipd(keys=["image", "mask_image", "target"], prob=0.8),
            # mn.transforms.RandAxisFlipd(keys=["image", "mask_image", "target"], prob=0.8),
            # mn.transforms.RandAxisFlipd(keys=["image", "mask_image", "target"], prob=0.8),
            custom.ChannelDropoutD(image_key="image", meta_key="params", max_ch=num_ch),
            custom.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                channel_wise=True,
            ),
            mn.transforms.LambdaD(
                keys=["image", "mask_image"], func=mn.transforms.SignalFillEmpty()
            ),
            mn.transforms.ResizeD(
                keys=["image", "mask_image"], spatial_size=(ptch, ptch, ptch)
            ),
        ]
    )

    train_data = mn.data.Dataset(train_dict, transform=train_transform)
    val_data = mn.data.Dataset(val_dict, transform=train_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
    )

    return train_loader, val_loader, train_transform