import monai as mn
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from random import shuffle, seed
import custom_cc
import custom
from monai.networks.nets import UNet
from tqdm import tqdm
import wandb
import argparse
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
import logging
import lesion

logging.getLogger("monai").setLevel(logging.ERROR)
logging.getLogger("monai.apps").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)


def print_tensor_info(tensor):
    print("Shape: ", tensor.shape)
    print("Min: ", tensor.min())
    print("Max: ", tensor.max())
    print("Mean: ", tensor.mean())
    print("Std: ", tensor.std())
    return tensor


def print_var_info(var):
    print(var)
    return var


def add_bg(tensor):
    return torch.cat([1 - tensor, tensor], dim=0)


def add_bg_mb(x):
    # return torch.cat([x, 1.-x.sum(0,keepdim=True)],dim=0)
    x = x - x.reshape(x.size(0), -1).min(1)[0].reshape(x.size(0), 1, 1, 1)
    x = x / x.reshape(x.size(0), -1).max(1)[0].reshape(x.size(0), 1, 1, 1)
    return torch.cat([1.0 - x.sum(0, keepdim=True), x], dim=0)


seed(786)


def get_synth_loaders(
    batch_size=1,
    device="cpu",
    lowres=False,
    ptch=128,
    pseudolabels=False,
):

    train_files = glob.glob(
        os.path.join(
            "/PATH/TO/OASIS/OAS*/OAS*_Freesurfer*/DATA/OAS*/mri/mni_1mm_healthy_symmetric.nii.gz"
        ),
    )

    train_dict = [
        {
            "label": f.replace("healthy_symmetric", "mb_labels"),
            "healthy": f,
        }
        for f in train_files
    ]

    shuffle(train_dict)

    train_dict, val_dict = (
        train_dict[:-100],
        train_dict[-100:],
    )

    if lesion:
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
            mn.transforms.LoadImageD(keys=["label", "healthy"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["label", "healthy"]),
            mn.transforms.SpacingD(
                keys=["label", "healthy"], pixdim=1 if not lowres else 2
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["label", "healthy"],
                spatial_size=(256, 256, 256) if not lowres else (128, 128, 128),
            ),
            mn.transforms.ToTensorD(
                dtype=float, keys=["label", "healthy"], device=device
            ),
            lesion.LesionPasteD(
                keys="label",
                new_keys=["seg"],
                label_list=train_label_list,
                mb_healthy=True,
                lesion_fading=True,
                lowres=lowres,
            ),
            mn.transforms.LambdaD(keys="label", func=add_bg_mb),
            custom_cc.SynthBloch(
                label_key="label",
                image_key="image",
                mpm_key="mpm",
                coreg_keys=["healthy"],
                num_ch=1,
                no_augs=True,
            ),
            mn.transforms.OneOf(
                transforms=[
                    custom_cc.RandomSkullStrip(
                        label_key="healthy", image_key=["image", "label", "mpm"]
                    ),
                    mn.transforms.IdentityD(keys="label"),
                ],
                weights=[0.3, 0.7],
            ),
            mn.transforms.Rand3DElasticD(
                keys=["image", "label", "seg", "mpm"],
                sigma_range=(5, 7),
                magnitude_range=(50, 150),
                rotate_range=15,
                shear_range=0.012,
                scale_range=0.15,
                padding_mode="zeros",
                prob=0.8,
                mode=["bilinear", "bilinear", "nearest", "bilinear"],
                allow_missing_keys=True,
            ),
            mn.transforms.LambdaD(
                keys=["image", "label", "seg", "mpm"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            mn.transforms.RandBiasFieldD(keys="image", prob=0.8),
            mn.transforms.RandAxisFlipd(
                keys=["image", "label", "seg", "mpm"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image", "label", "seg", "mpm"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image", "label", "seg", "mpm"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.RandGaussianNoiseD(keys="image", prob=0.8),
            mn.transforms.RandSimulateLowResolutionD(
                keys=["image"], prob=0.6, zoom_range=(0.3, 0.9)
            ),
            mn.transforms.RandSpatialCropD(
                keys=["image", "label", "seg", "mpm"],
                roi_size=(ptch, ptch, ptch),
                random_size=False,
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeD(
                keys=["image", "label", "seg", "mpm"],
                spatial_size=(ptch, ptch, ptch),
                allow_missing_keys=True,
            ),
            mn.transforms.ThresholdIntensityD(
                keys=["seg"],
                threshold=0.5,
                above=True,
                cval=0,
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(dtype=torch.float32, keys="image"),
            mn.transforms.ToTensorD(
                dtype=torch.float32, keys=["seg"], allow_missing_keys=True
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

    return train_loader, val_loader


def get_loaders(
    batch_size=1,
    device="cpu",
    lowres=False,
    ptch=128,
    baseline=False,
    gmm=False,
    pseudolabels=False,
):

    train_label_list = list(
        np.loadtxt("/PATH/TO/ATLAS/atlas_train.txt", dtype=str)
    )
    val_label_list = list(
        np.loadtxt("/PATH/TO/ATLAS/atlas_val.txt", dtype=str)
    )

    datadir = "/PATH/TO/ATLAS-MPM/"

    if baseline:
        train_dict = [
            {
                "seg": f.replace("1mm_sub", "sub"),
                "label": f.replace("_label-L_desc-T1lesion_mask", "_T1w").replace(
                    "1mm_sub", "sub"
                ),
                "mask": os.path.join(
                    datadir,
                    f.split("/")[-1]
                    .replace("1mm_sub", "sub")
                    .replace("_label-L_desc-T1lesion_mask", "_T1w_mask_fixed"),
                ),
            }
            for f in train_label_list
        ]
        val_dict = [
            {
                "seg": f.replace("1mm_sub", "sub"),
                "label": f.replace("_label-L_desc-T1lesion_mask", "_T1w").replace(
                    "1mm_sub", "sub"
                ),
                "mask": os.path.join(
                    datadir,
                    f.split("/")[-1]
                    .replace("1mm_sub", "sub")
                    .replace("_label-L_desc-T1lesion_mask", "_T1w_mask_fixed"),
                ),
            }
            for f in val_label_list
        ]

        if pseudolabels:
            pseudo_label_list = glob.glob(
                "/PATH/TO/PLORAS/1mm_*_lesion.nii.gz"
            )
            pseudo_dict = [
                {
                    "seg": f,
                    "label": f.replace("_lesion", "_T1w"),
                    "mask": f.replace("_lesion", "_mask"),
                }
                for f in pseudo_label_list
            ]
            train_dict += pseudo_dict

    else:
        train_label_list = [
            os.path.join(datadir, lst.split("/")[-1].replace("1mm_sub", "sub"))
            for lst in train_label_list
        ]
        val_label_list = [
            os.path.join(datadir, lst.split("/")[-1].replace("1mm_sub", "sub"))
            for lst in val_label_list
        ]

        train_dict = [
            {
                "label": [
                    f.replace("_label-L_desc-T1lesion_mask", "_PD"),
                    f.replace("_label-L_desc-T1lesion_mask", "_R1"),
                    f.replace("_label-L_desc-T1lesion_mask", "_R2s"),
                    f.replace("_label-L_desc-T1lesion_mask", "_MT"),
                ],
                "seg": f,
                "mask": f.replace("_label-L_desc-T1lesion_mask", "_T1w_mask_fixed"),
            }
            for f in train_label_list
        ]
        val_dict = [
            {
                "label": [
                    f.replace("_label-L_desc-T1lesion_mask", "_PD"),
                    f.replace("_label-L_desc-T1lesion_mask", "_R1"),
                    f.replace("_label-L_desc-T1lesion_mask", "_R2s"),
                    f.replace("_label-L_desc-T1lesion_mask", "_MT"),
                ],
                "seg": f,
                "mask": f.replace("_label-L_desc-T1lesion_mask", "_T1w_mask_fixed"),
            }
            for f in val_label_list
        ]

        if pseudolabels:
            pseudo_label_list = glob.glob(
                "/PATH/TO/PLORAS/1mm_*_lesion.nii.gz"
            )
            pseudo_dict = [
                {
                    "label": [
                        f.replace("_lesion", "_PD"),
                        f.replace("_lesion", "_R1"),
                        f.replace("_lesion", "_R2s"),
                        f.replace("_lesion", "_MT"),
                    ],
                    "seg": f,
                    "mask": f.replace("_lesion", "_mask"),
                }
                for f in pseudo_label_list
            ]
            train_dict += pseudo_dict

    print(f"train_dict: {len(train_dict)}, val_dict: {len(val_dict)}")
    print(f"train_dict[0]: {train_dict[0]}")

    train_transform = mn.transforms.Compose(
        transforms=[
            # mn.transforms.LambdaD(keys=["label", "seg"], func=print_var_info),
            mn.transforms.LoadImageD(
                keys=["label", "seg", "mask"], image_only=True, allow_missing_keys=True
            ),
            # mn.transforms.LambdaD(keys=["label", "seg"], func=print_tensor_info),
            mn.transforms.EnsureChannelFirstD(
                keys=["label", "seg", "mask"], allow_missing_keys=True
            ),
            mn.transforms.LambdaD(keys="seg", func=add_bg),
            mn.transforms.OrientationD(
                keys=["label", "seg", "mask"], axcodes="RAS", allow_missing_keys=True
            ),
            mn.transforms.SpacingD(
                keys=["label", "seg", "mask"],
                pixdim=1 if not lowres else 2,
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["label", "seg", "mask"],
                spatial_size=(256, 256, 256) if not lowres else (128, 128, 128),
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(
                dtype=float,
                keys=["label", "seg", "mask"],
                device=device,
                allow_missing_keys=True,
            ),
            mn.transforms.LambdaD(
                keys=["label", "seg", "mask"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            (
                custom.GMMAugmentD(image_key="label", mask_key="mask")
                if gmm
                else mn.transforms.IdentityD(keys=["mask"], allow_missing_keys=True)
            ),
            # mn.transforms.LambdaD(keys=["label", "seg"], func=print_tensor_info),
            # custom_cc.SynthBloch(label_key="label", image_key="image", mpm_key="mpm", coreg_keys=["seg"], num_ch=1, label_source="freesurfer",
            #                         elastic_nodes=5 if lowres else 10, gmm_fwhm=0.01, bias=3 if lowres else 7,
            #                         resolution=4 if lowres else 8, gfactor=3 if lowres else 5, no_augs=no_augs, use_real_mpms=True),
            (
                custom_cc.SynthBloch(
                    label_key="label",
                    image_key="image",
                    mpm_key="mpm",
                    coreg_keys=["seg", "mask"],
                    num_ch=1,
                    label_source="freesurfer",
                    no_augs=True,
                    use_real_mpms=True,
                )
                if not baseline
                else mn.transforms.CopyItemsD(
                    keys=["label"], times=2, names=["image", "mpm"]
                )
            ),
            mn.transforms.Rand3DElasticD(
                keys=["image", "label", "seg", "mpm", "mask"],
                sigma_range=(5, 7),
                magnitude_range=(50, 150),
                rotate_range=15,
                shear_range=0.012,
                scale_range=0.15,
                padding_mode="zeros",
                prob=0.8,
                mode=["bilinear", "bilinear", "nearest", "bilinear", "nearest"],
                allow_missing_keys=True,
            ),
            mn.transforms.LambdaD(
                keys=["image", "label", "seg", "mpm", "mask"],
                func=mn.transforms.SignalFillEmpty(),
                allow_missing_keys=True,
            ),
            (
                mn.transforms.MaskIntensityD(
                    keys=["image", "label", "seg", "mpm"], mask_key="mask"
                )
                if not baseline
                else mn.transforms.IdentityD(keys=["mask"], allow_missing_keys=True)
            ),
            # mn.transforms.HistogramNormalizeD(keys="image"),
            # mn.transforms.RandHistogramShiftD(keys="image", prob=0.8),
            mn.transforms.RandBiasFieldD(keys="image", prob=0.8),
            # mn.transforms.RandAdjustContrastD(keys="image", prob=0.8),
            mn.transforms.RandAxisFlipd(
                keys=["image", "label", "seg", "mpm", "mask"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image", "label", "seg", "mpm", "mask"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.RandAxisFlipd(
                keys=["image", "label", "seg", "mpm", "mask"],
                prob=0.8,
                allow_missing_keys=True,
            ),
            mn.transforms.NormalizeIntensityD(
                keys="image", nonzero=False, channel_wise=True
            ),
            mn.transforms.RandGaussianNoiseD(keys="image", prob=0.8),
            mn.transforms.RandSimulateLowResolutionD(
                keys=["image"], prob=0.6, zoom_range=(0.3, 0.9)
            ),
            mn.transforms.RandSpatialCropD(
                keys=["image", "label", "seg", "mpm", "mask"],
                roi_size=(ptch, ptch, ptch),
                random_size=False,
                allow_missing_keys=True,
            ),
            mn.transforms.ResizeD(
                keys=["image", "label", "seg", "mpm", "mask"],
                spatial_size=(ptch, ptch, ptch),
                allow_missing_keys=True,
            ),
            # mn.transforms.LambdaD(keys=["label", "seg"], func=print_tensor_info),
            mn.transforms.ThresholdIntensityD(
                keys=["seg", "mask"],
                threshold=0.5,
                above=True,
                cval=0,
                allow_missing_keys=True,
            ),
            mn.transforms.ToTensorD(dtype=torch.float32, keys="image"),
            mn.transforms.ToTensorD(
                dtype=torch.float32, keys=["seg", "mask"], allow_missing_keys=True
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

    return train_loader, val_loader


def compute_dice(y_pred, y, eps=1e-8):
    y_pred = torch.flatten(y_pred)
    y = torch.flatten(y)
    y = y.float()
    intersect = (y_pred * y).sum(-1)
    denominator = (y_pred * y_pred).sum(-1) + (y * y).sum(-1)
    return 2 * (intersect / denominator.clamp(min=eps))


def run_model(args, device, train_loader, val_loader):
    dim = 2 if args.twod else 3
    model = UNet(
        spatial_dims=dim,
        in_channels=1,
        out_channels=6 if args.synth else 2,
        channels=[32, 64, 128, 256, 320, 320],
        strides=[2, 2, 2, 2, 2],
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=1,
        act="PRELU",
        norm="INSTANCE",
        dropout=args.dropout,
        bias=True,
        adn_ordering="NDA",
    ).to(device)

    if args.resume or args.resume_best:
        ckpts = glob.glob(
            os.path.join(
                args.logdir,
                args.name,
                "checkpoint.pt" if args.resume else "checkpoint_best.pt",
            )
        )
        if len(ckpts) == 0:
            args.resume = False
            print("\nNo checkpoints found. Beginning from epoch #0")
        else:
            checkpoint = torch.load(ckpts[0], map_location=device)
            print(
                "\nResuming from epoch #{} with WandB ID {}".format(
                    checkpoint["epoch"], checkpoint["wandb"]
                )
            )
    print()

    if args.reset_wandb:
        wandb.init(
            project="synthbloch",
            entity="atlas-ploras",
            save_code=True,
            name=args.name,
            settings=wandb.Settings(start_method="fork"),
            resume=None,
            id=None,
        )
    else:
        try:
            wandb.init(
                project="synthbloch",
                entity="atlas-ploras",
                save_code=True,
                name=args.name,
                settings=wandb.Settings(start_method="fork"),
                resume="must" if args.resume else None,
                id=checkpoint["wandb"] if args.resume or args.resume_best else None,
            )
            if not args.resume or args.resume_best:
                wandb.config.update(args)
        except:
            print("WandB resume failed. Treating as new run.")
            wandb.init(
                project="synthbloch",
                entity="atlas-ploras",
                save_code=True,
                name=args.name,
                settings=wandb.Settings(start_method="fork"),
                resume=None,
                id=None,
            )
    wandb.watch(model)

    crit = mn.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=False,
        sigmoid=False,
        softmax=True,
        other_act=None,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
        smooth_nr=1e-05,
        smooth_dr=1e-05,
        batch=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id

    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return self.epoch

    class Metric:
        def __init__(self, metric):
            self.metric = metric

        def state_dict(self):
            return self.metric

    lab_dict = (
        {
            0: "Background",
            1: "Gray Matter",
            2: "Gray/White PV",
            3: "White Matter",
            4: "CSF",
            5: "Stroke lesion",
        }
        if args.synth
        else {0: "Background", 1: "Stroke lesion"}
    )

    try:
        opt = torch.optim.AdamW(
            model.parameters(), args.lr, foreach=torch.cuda.is_available()
        )
    except:
        opt = torch.optim.AdamW(model.parameters(), args.lr)
    # Try to load most recent weight
    if args.resume or args.resume_best:
        model.load_state_dict(checkpoint["net"])
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"] + 1
        metric_best = checkpoint["metric"]
    else:
        start_epoch = 0
        metric_best = 0

    # override learning rate stuff
    def lambda1(epoch):
        return (1 - (epoch + start_epoch) / args.epochs) ** 0.9

    for param_group in opt.param_groups:
        param_group["lr"] = lambda1(0) * args.lr
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])

    train_iter = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        if args.amp:
            ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
            scaler = torch.cuda.amp.GradScaler()
        else:
            ctx = nullcontext()
        progress_bar = tqdm(range(args.epoch_length), total=args.epoch_length, ncols=60)
        progress_bar.set_description(f"[Training] Epoch {epoch}")
        if train_iter is None:
            train_iter = iter(train_loader)
        for step in progress_bar:
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            images = batch["image"].to(device)
            labels = batch["seg"].to(device)
            opt.zero_grad(set_to_none=True)
            with ctx:
                logits = model(images)
                if logits.shape[1] != labels.shape[1] and labels.shape[1] == 2:
                    logits = logits[:,[0,-1]] # extract background + lesion
                loss = crit(logits, labels)
                # print(f"\n\nLoss: {loss}")
                # print(f"Images: {images.shape} min: {images.min()} max: {images.max()}")
                # print(f"Labels: {labels.shape} min: {labels.min()} max: {labels.max()}")
                # print(f"Logits: {logits.shape} min: {logits.min()} max: {logits.max()}")
                # print(f"Probs: {torch.softmax(logits, dim=1).shape} min: {torch.softmax(logits, dim=1).min()} max: {torch.softmax(logits, dim=1).max()}")
            if args.stop_nans:
                assert not loss.isnan().sum(), "NaN found in loss!"
            if loss.isnan().sum() != 0:
                print("NaN found in loss!")
            else:
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    opt.step()
                epoch_loss += loss.sum().item()
                wandb.log({"train/loss": loss.sum().item()})
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            if args.test_run:
                break

        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            dice_metric = []
            plots = []
            ctx = (
                torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
                if args.amp
                else nullcontext()
            )
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=60)
            progress_bar.set_description(f"[Validation] Epoch {epoch}")
            with torch.no_grad():
                for val_step, batch in progress_bar:
                    images = batch["image"].to(device)
                    labels = batch["seg"].to(device)
                    opt.zero_grad(set_to_none=True)
                    with ctx:
                        logits = model(images)
                        probs = torch.softmax(logits, dim=1)
                        dice_metric.append(
                            compute_dice(y_pred=probs[:, -1], y=labels[:, -1])
                            .mean()
                            .cpu()
                            .item()
                        )  # lesion only
                    if val_step < 5:
                        plots.append(
                            wandb.Image(
                                images[0, 0, ..., images.size(-1) // 2].cpu().float(),
                                masks={
                                    "predictions": {
                                        "mask_data": probs[0]
                                        .argmax(0)
                                        .cpu()[..., probs.size(-1) // 2],
                                        "class_labels": lab_dict,
                                    },
                                    "ground truth": {
                                        "mask_data": labels[0]
                                        .argmax(0)
                                        .cpu()[..., labels.size(-1) // 2],
                                        "class_labels": lab_dict,
                                    },
                                },
                            )
                        )
                    elif val_step == 5:
                        wandb.log({"val/examples": plots})
                        if args.test_run:
                            break
            metric = np.nanmean(dice_metric)
            wandb.log({"val/dice": metric})

            if metric > metric_best:
                metric_best = metric
                torch.save(
                    {
                        "net": model.state_dict(),
                        "opt": opt.state_dict(),
                        "lr": lr_scheduler.state_dict(),
                        "wandb": WandBID(wandb.run.id).state_dict(),
                        "epoch": Epoch(epoch).state_dict(),
                        "metric": Metric(metric_best).state_dict(),
                    },
                    os.path.join(
                        args.logdir, args.name, "checkpoint_best.pt".format(epoch)
                    ),
                )
            torch.save(
                {
                    "net": model.state_dict(),
                    "opt": opt.state_dict(),
                    "lr": lr_scheduler.state_dict(),
                    "wandb": WandBID(wandb.run.id).state_dict(),
                    "epoch": Epoch(epoch).state_dict(),
                    "metric": Metric(metric_best).state_dict(),
                },
                os.path.join(args.logdir, args.name, "checkpoint.pt".format(epoch)),
            )


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training."
    )
    parser.add_argument(
        "--epoch_length", type=int, default=100, help="Number of iterations per epoch."
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout ratio.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--val_interval", type=int, default=2, help="Validation interval."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--patch", type=int, default=128, help="Patch size for cropping."
    )
    parser.add_argument("-a", "--amp", default=False, action="store_true")
    parser.add_argument(
        "--logdir", type=str, default="./", help="Path to saved outputs"
    )
    parser.add_argument("-dbg", "--debug", default=False, action="store_true")
    parser.add_argument("-res", "--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument(
        "--twod",
        default=False,
        action="store_true",
        help="Train in 2D. Useful for quick debugging.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. If not specified then will check for CUDA.",
    )
    parser.add_argument(
        "--test_run",
        default=False,
        action="store_true",
        help="Run single iteration per epoch for quick debug.",
    )
    parser.add_argument(
        "--lowres",
        default=False,
        action="store_true",
        help="Train with un-cropped 2D images.",
    )
    parser.add_argument(
        "--stop_nans",
        action="store_true",
        help="Hard stop in presence of NaNs. Otherwise will give warning and skip.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline model - use real images only.",
    )
    parser.add_argument(
        "--gmm",
        action="store_true",
        help="Use Gaussian Mixture Model for augmentation.",
    )
    parser.add_argument(
        "--pseudolabels",
        action="store_true",
        help="Use pseudolabel data.",
    )
    parser.add_argument(
        "--synth",
        action="store_true",
        help="Use synthseg-like qMRI data.",
    )
    parser.add_argument(
        "--mix",
        action="store_true",
        help="Mix data with real (baseline) data.",
    )
    parser.add_argument(
        "--reset_wandb",
        action="store_true",
        help="Reset WandB run.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.synth:
        train_loader, val_loader = get_synth_loaders(
            args.batch_size, device, args.lowres, args.patch, args.pseudolabels
        )
    else:
        train_loader, val_loader = get_loaders(
            args.batch_size,
            device,
            args.lowres,
            args.patch,
            args.baseline,
            args.gmm,
            args.pseudolabels,
        )

    if args.mix:
        import random
        train_rl_loader, val_rl_loader = get_loaders(args.batch_size, device, args.lowres, args.patch, baseline=True, gmm=False, pseudolabels=False)
        def chunk(indices, size):
            return torch.split(torch.tensor(indices), size)

        class MyBatchSampler(torch.utils.data.Sampler):
            def __init__(self, a_indices, b_indices, batch_size): 
                self.a_indices = a_indices
                self.b_indices = b_indices
                self.batch_size = batch_size
            
            def __iter__(self):
                random.shuffle(self.a_indices)
                random.shuffle(self.b_indices)
                a_batches  = chunk(self.a_indices, self.batch_size)
                b_batches = chunk(self.b_indices, self.batch_size)
                all_batches = list(a_batches + b_batches)
                all_batches = [batch.tolist() for batch in all_batches]
                random.shuffle(all_batches)
                return iter(all_batches)
        
        new_dataset = torch.utils.data.ConcatDataset((train_loader.dataset, train_rl_loader.dataset))
        a_len = train_loader.__len__()
        ab_len = a_len + train_rl_loader.__len__()
        print(f"a_len: {a_len}, ab_len: {ab_len}")
        a_indices = list(range(a_len))
        b_indices = list(range(a_len, ab_len))
        batch_sampler = MyBatchSampler(a_indices, b_indices, train_loader.batch_size)
        train_loader = torch.utils.data.DataLoader(new_dataset,  batch_sampler=batch_sampler)

    return args, device, train_loader, val_loader


def main():
    args, device, train_loader, val_loader = set_up()
    if args.debug:
    #     saver1 = mn.transforms.SaveImage(
    #         output_dir=os.path.join(args.logdir, args.name, "debug-val"),
    #         output_postfix="img",
    #         separate_folder=False,
    #     )
    #     saver2 = mn.transforms.SaveImage(
    #         output_dir=os.path.join(args.logdir, args.name, "debug-val"),
    #         output_postfix="label",
    #         separate_folder=False,
    #     )
    #     for i, batch in enumerate(val_loader):
    #         if i > 9:
    #             break
    #         else:
    #             print(
    #                 "Image: ",
    #                 batch["image"].shape,
    #                 "min={}".format(batch["image"].min()),
    #                 "max={}".format(batch["image"].max()),
    #             )
    #             print(
    #                 "seg: ",
    #                 batch["seg"].shape,
    #                 "min={}".format(batch["seg"].min()),
    #                 "max={}".format(batch["seg"].max()),
    #             )
    #             saver1(
    #                 torch.Tensor(batch["image"][0].cpu().float()),
    #             )
    #             saver2(
    #                 torch.Tensor(
    #                     torch.argmax(batch["seg"][0], dim=0)[None].cpu().float()
    #                 ),
    #             )
        saver1 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug-train"),
            output_postfix="img",
            separate_folder=False,
        )
        saver2 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug-train"),
            output_postfix="label",
            separate_folder=False,
        )
        for i, batch in enumerate(train_loader):
            if i > 9:
                break
            else:
                print(
                    "Image: ",
                    batch["image"].shape,
                    "min={}".format(batch["image"].min()),
                    "max={}".format(batch["image"].max()),
                )
                print(
                    "seg: ",
                    batch["seg"].shape,
                    "min={}".format(batch["seg"].min()),
                    "max={}".format(batch["seg"].max()),
                )
                saver1(
                    torch.Tensor(batch["image"][0].cpu().float()),
                )
                saver2(
                    torch.Tensor(
                        torch.argmax(batch["seg"][0], dim=0)[None].cpu().float()
                    ),
                )
        print("Debug finished and samples saved.")
        exit()
    run_model(args, device, train_loader, val_loader)


if __name__ == "__main__":
    main()