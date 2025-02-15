import monai as mn
import glob
import os
from torch.utils.data import DataLoader
import torch
from random import shuffle, seed
import custom_cc
import custom
import numpy as np
import lesion

seed(786)


def add_bg(x):
    # return torch.cat([x, 1.-x.sum(0,keepdim=True)],dim=0)
    x = x - x.reshape(x.size(0), -1).min(1)[0].reshape(x.size(0), 1, 1, 1)
    x = x / x.reshape(x.size(0), -1).max(1)[0].reshape(x.size(0), 1, 1, 1)
    return torch.cat([1.0 - x.sum(0, keepdim=True), x], dim=0)


def get_loaders(
    batch_size=1,
    device="cpu",
    lowres=False,
    num_ch=1,
    local_paths=False,
    mni=False,
    use_lesion=False,
    pseudolabels=False,
):
    if local_paths:
        train_files = glob.glob(
            os.path.join(
                "/PATH/TO/OASIS/OAS*/OAS*_Freesurfer*/DATA/OAS*/mri/mni_1mm_healthy_symmetric.nii.gz"
            ),
        )
    else:
        train_files = glob.glob(
            os.path.join(
                "/PATH/TO/OASIS/OAS*/OAS*_Freesurfer*/DATA/OAS*/mri/mni_1mm_healthy_symmetric.nii.gz"
            ),
        )
    if (
        not mni
    ):  # removed restriction to MNI images - this way we can also infer for neck
        train_files = [f.replace("mni_1mm_", "") for f in train_files]
    train_dict = [
        {
            "label": f.replace("healthy_symmetric", "mb_labels"),
            "freesurf": f,
        }
        for f in train_files
    ]

    shuffle(train_dict)

    train_dict, val_dict = (
        train_dict[:-100],
        train_dict[-100:],
    )

    if use_lesion:
        train_label_list = list(
            np.loadtxt("/PATH/TO/ATLAS/atlas_train.txt", dtype=str)
        )
        val_label_list = list(
            np.loadtxt("/PATH/TO/ATLAS/atlas_val.txt", dtype=str)
        )
    if pseudolabels:
        train_label_list += glob.glob(
            "/PATH/TO/PLORAS/1mm_*_lesion.nii.gz"
        )

    ptch = 96 if lowres else 192

    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.CopyItemsD(keys=["label"], names=["path"]),
            mn.transforms.LoadImageD(keys=["label", "freesurf"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["label", "freesurf"]),
            mn.transforms.OrientationD(keys=["label", "freesurf"], axcodes="RAS"),
            mn.transforms.LambdaD(keys="label", func=add_bg),
            mn.transforms.SpacingD(
                keys=["label", "freesurf"], pixdim=2 if lowres else 1
            ),
            mn.transforms.ToTensorD(
                dtype=float, keys=["label", "freesurf"], device=device
            ),
            mn.transforms.CopyItemsD(keys=["label"], names=["healthy"]),
            (
                lesion.LesionPasteD(
                    keys="label",
                    new_keys=["seg"],
                    label_list=train_label_list,
                    mb_healthy=True,
                    lesion_fading=True,
                    lowres=lowres,
                )
                if use_lesion
                else mn.transforms.LambdaD(keys="label", func=lambda x: x)
            ),
            custom_cc.SynthBloch(
                label_key="label",
                image_key="image",
                mpm_key="mpm",
                coreg_keys=["freesurf"],
                num_ch=num_ch,
            ),
            mn.transforms.OneOf(
                transforms=[
                    custom_cc.RandomSkullStrip(
                        label_key="freesurf", image_key=["image", "label", "mpm"]
                    ),
                    mn.transforms.IdentityD(keys="label"),
                ],
                weights=[0.3, 0.7],
            ),
            mn.transforms.RandSpatialCropD(
                keys=["image", "label", "mpm"], roi_size=ptch, random_size=False
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["image", "label", "mpm"], spatial_size=(ptch, ptch, ptch)
            ),
            mn.transforms.RandAxisFlipd(keys=["image", "label", "mpm"], prob=0.8),
            mn.transforms.RandAxisFlipd(keys=["image", "label", "mpm"], prob=0.8),
            mn.transforms.RandAxisFlipd(keys=["image", "label", "mpm"], prob=0.8),
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                channel_wise=True,
            ),
            custom.ChannelDropoutD(image_key="image", meta_key="params", max_ch=num_ch),
            mn.transforms.LambdaD(
                keys=["image", "label", "mpm"], func=mn.transforms.SignalFillEmpty()
            ),
            mn.transforms.ResizeD(
                keys=["image", "label", "mpm"], spatial_size=(ptch, ptch, ptch)
            ),
            mn.transforms.ToTensorD(
                dtype=torch.float32, keys=["image", "label", "mpm"]
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