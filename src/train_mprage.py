import glob
import os

from model import UNet
from generative.metrics import SSIMMetric
from generative.losses import PerceptualLoss
from torch.utils.data import DataLoader
import torch
from random import shuffle, seed
from preprocess import get_loaders
from loss import BarronLoss, BaurLoss
from tqdm import tqdm
import monai as mn
import custom_cc

import wandb
import argparse
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
from torchvision.utils import make_grid
import logging

logging.getLogger("monai").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)


def activate(x):
    # apply activation function to match postproc in nitorch qmri
    x[:, 0] = x[:, 0].exp()  # pd = e^f(x)
    x[:, 1] = x[:, 1].exp()  # r1 = e^f(x)
    x[:, 2] = x[:, 2].exp()  # r2 = e^f(x)
    x[:, 3] = x[:, 3].neg().exp().add(1).reciprocal().mul(100)  # mt = 100/(1+e^-f(x))
    x[:, 1] = x[:, 1] / 10
    x[:, 2] = x[:, 2] / 10
    return x


seed(786)


def rescale_mpm(mpm):
    pd = mpm[0]
    r1 = mpm[1]
    r2s = mpm[2]
    mt = mpm[3]
    r1 = r1 * 10
    r2s = r2s * 10
    return torch.stack([pd, r1, r2s, mt], dim=0)


def get_loaders(
    batch_size=1,
    device="cpu",
    lowres=False,
    only_stroke=False,
):
    if only_stroke:
        train_files = glob.glob(
            os.path.join("/PATH/TO/STROKE_QMRI/*/masked_pd.nii"),
        )
    else:
        train_files = glob.glob(
            os.path.join("/PATH/TO/STROKE_QMRI/*/masked_pd.nii"),
        ) + glob.glob(
            os.path.join("/PATH/TO/HEALTHY_QMRI/*/masked_pd.nii"),
        )

    train_dict = [
        {
            "label": [
                f,
                f.replace("masked_pd.nii", "masked_r1.nii"),
                f.replace("masked_pd.nii", "masked_r2s.nii"),
                f.replace("masked_pd.nii", "masked_mt.nii"),
            ],
            "mask": f.replace("masked_pd.nii", "mask.nii"),
        }
        for f in train_files
    ]

    shuffle(train_dict)

    train_dict, val_dict = (
        train_dict[:-5],
        train_dict[-5:],
    )

    ptch = 96 if lowres else 192

    train_transform = mn.transforms.Compose(
        transforms=[
            mn.transforms.CopyItemsD(keys=["label"], names=["path"]),
            mn.transforms.LoadImageD(keys=["label", "mask"], image_only=True),
            mn.transforms.EnsureChannelFirstD(keys=["label", "mask"]),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["label", "mask"], spatial_size=(256, 256, 256)
            ),
            mn.transforms.LambdaD(keys=["label"], func=mn.transforms.SignalFillEmpty()),
            # mn.transforms.LambdaD(keys=["label"], func=rescale_mpm),
            custom_cc.ClipPercentilesD(
                keys=["label"],
                lower=0.5,
                upper=99.5,
            ),  # just clip extreme values, don't rescale
            mn.transforms.LambdaD(keys=["label"], func=mn.transforms.SignalFillEmpty()),
            mn.transforms.OrientationD(keys=["label", "mask"], axcodes="RAS"),
            mn.transforms.SpacingD(keys=["label", "mask"], pixdim=2 if lowres else 1),
            mn.transforms.ToTensorD(
                dtype=torch.float32, keys=["label", "mask"], device=device
            ),
            custom_cc.SynthBloch(
                label_key="label",
                image_key="image",
                mpm_key="mpm",
                coreg_keys=["mask"],
                num_ch=1,
                label_source="freesurfer",
                no_augs=True,
                use_real_mpms=True,
                sequence=["mprage-t1"],
            ),
            mn.transforms.Rand3DElasticD(
                keys=["image", "mpm", "mask"],
                sigma_range=(5, 7),
                magnitude_range=(50, 150),
                prob=1,
                padding_mode="zeros",
                mode=["bilinear", "bilinear", "nearest"],
            ),
            mn.transforms.RandAffineD(
                keys=["image", "mpm", "mask"],
                prob=1,
                shear_range=(0.01, 0.2),
                mode=["bilinear", "bilinear", "nearest"],
                padding_mode="zeros",
            ),
            mn.transforms.RandBiasFieldD(
                keys=["image"], coeff_range=(0.1, 0.3), prob=0.8
            ),
            mn.transforms.RandGibbsNoiseD(keys=["image"], prob=0.8, alpha=(0.1, 0.8)),
            mn.transforms.RandRicianNoiseD(
                keys=["image"],
                prob=0.8,
                mean=0.0,
                std=0.5,
                relative=True,
                sample_std=True,
            ),
            mn.transforms.RandSpatialCropD(
                keys=["image", "mpm", "mask"], roi_size=ptch, random_size=False
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["image", "mpm", "mask"], spatial_size=(ptch, ptch, ptch)
            ),
            # mn.transforms.RandAxisFlipd(keys=["image","mpm","mask"], prob=0.8),
            # mn.transforms.RandAxisFlipd(keys=["image","mpm","mask"], prob=0.8),
            # mn.transforms.RandAxisFlipd(keys=["image","mpm","mask"], prob=0.8),
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                channel_wise=True,
            ),
            mn.transforms.LambdaD(
                keys=["image", "mpm"], func=mn.transforms.SignalFillEmpty()
            ),
            mn.transforms.ResizeD(
                keys=["image", "mpm", "mask"], spatial_size=(ptch, ptch, ptch)
            ),
            mn.transforms.ToTensorD(
                dtype=torch.float32, keys=["image", "mpm", "mask", "params"]
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


def run_model(args, device, train_loader, val_loader, train_transform):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=[24, 48, 96, 192, 384],
        strides=[2, 2, 2, 2],
        dropout=0.1,
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
        act="GELU",
        norm="INSTANCE",
        adn_ordering="NDA",
        upsample=args.upsample,
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

    if args.reset_training:
        wandb_id = None
    elif args.resume or args.resume_best:
        wandb_id = checkpoint["wandb"]
    else:
        wandb_id = None

    wandb.init(
        project="synthbloch",
        entity="atlas-ploras",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if wandb_id is not None else None,
        id=wandb_id,
    )
    if not args.resume or args.resume_best:
        wandb.config.update(args)
    wandb.watch(model)

    if args.loss == "l1":
        l1_loss = torch.nn.L1Loss(reduction="none")
    elif args.loss == "l2":
        l1_loss = torch.nn.MSELoss(reduction="none")
    elif args.loss == "huber":
        l1_loss = torch.nn.HuberLoss(reduction="none")
    elif args.loss == "baur":
        l1_loss = BaurLoss(reduction="none")
    elif args.loss == "barron":
        l1_loss = BarronLoss(reduction="none")
    else:
        raise ValueError(
            "Loss '{}' not found. Please use train.py --help to see available options.".format(
                args.loss
            )
        )
    loss_perceptual = PerceptualLoss(
        spatial_dims=3, network_type="medicalnet_resnet50_23datasets", is_fake_3d=False
    ).to(device)
    perceptual_weight = 0.1
    ssim = SSIMMetric(spatial_dims=3, data_range=1.0, reduction="mean", kernel_size=5)

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

    try:
        opt = torch.optim.AdamW(
            model.parameters(), args.lr, fused=torch.cuda.is_available()
        )
    except:
        opt = torch.optim.AdamW(model.parameters(), args.lr)


    

    # Try to load most recent weight
    if args.resume or args.resume_best:
        model.load_state_dict(
            checkpoint["net"], strict=False
        )  # strict False in case of switch between subpixel and transpose
        if not args.reset_training:
            opt.load_state_dict(checkpoint["opt"])
            start_epoch = checkpoint["epoch"] + 1
            metric_best = checkpoint["metric"]
        else:
            start_epoch = 0
            metric_best = 0
    else:
        start_epoch = 0
        metric_best = 0

    def lambda1(epoch):
        return (1 - (epoch + start_epoch) / args.epochs) ** 0.9
    lr_scheduler = LambdaLR(opt, lr_lambda=[lambda1])

    train_len = len(train_loader)
    val_len = len(val_loader)

    print()
    print("Beginning training with dataset of size:")
    print("TRAIN: {}".format(int(len(train_loader) * args.batch_size)))
    print("VAL: {}".format(int(len(val_loader))))
    print()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    def normalise_mpm(target, pred, clip=False):
        pred = torch.stack(
            [pred[:, i] / target[:, i].max() for i in range(pred.size(1))], dim=1
        )
        target = torch.stack(
            [target[:, i] / target[:, i].max() for i in range(pred.size(1))], dim=1
        )
        if clip:
            target = torch.clamp(target, 0, 1)
            pred = torch.clamp(pred, 0, 1)
        return target, pred

    def rescale_fwd(pred, ref, clip=False):
        ref_min = ref[:, 0]
        ref_max = ref[:, 1]
        pred = (pred - ref_min) / (ref_max - ref_min)
        if clip:
            pred = torch.clamp(pred, 0, 1)
        return pred

    train_iter = None
    for epoch in range(start_epoch, args.epochs):
        if args.debug:
            saver1 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="img",
                separate_folder=False,
                print_log=False,
            )
            saver2 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="pred",
                separate_folder=False,
                print_log=False,
            )
            saver3 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="recon",
                separate_folder=False,
                print_log=False,
            )
        crit = l1_loss if epoch > args.l2_epochs else torch.nn.MSELoss(reduction="none")
        model.train()
        epoch_loss = 0
        step_deficit = -1e-7
        if args.amp:
            ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
            scaler = torch.cuda.amp.GradScaler()
        else:
            ctx = nullcontext()
        progress_bar = tqdm(range(args.epoch_length), total=args.epoch_length, ncols=90)
        progress_bar.set_description(f"Epoch {epoch}")
        if train_iter is None:
            train_iter = iter(train_loader)

        torch.autograd.set_detect_anomaly(args.anomaly)

        for step in progress_bar:
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            images = batch["image"].to(device)
            target = batch["mpm"].to(images)
            mask = batch["mask"].to(device)
            params = batch["params"].to(images)
            if (params > 1).sum().item() > 0:
                print("\nParams too large:")
                print("File: ", batch["path"])
                print(params)
            opt.zero_grad(set_to_none=True)

            if args.debug and step < 5:
                saver1(torch.Tensor(images[0].cpu().float()))

            with ctx:
                reconstruction = activate(model(images))
                if args.debug and step < 5:
                    recon_ = custom_cc.forward_model(reconstruction, params, 1)
                    if args.mask:
                        saver2(torch.Tensor((reconstruction * mask)[0].cpu().float()))
                        saver3(torch.Tensor((recon_ * mask)[0].cpu().float()))
                    else:
                        saver2(torch.Tensor(reconstruction[0].cpu().float()))
                        saver3(torch.Tensor(recon_[0].cpu().float()))
                target, reconstruction = normalise_mpm(
                    target, reconstruction, clip=False
                )
                recons_loss = crit(reconstruction, target)
                if args.mask:
                    recons_loss = (
                        mask * recons_loss
                    )  # mask to only calculate foreground
                    reconstruction = mask * reconstruction
                    target = mask * target
                recons_loss = recons_loss.mean()
                if epoch > args.l2_epochs:
                    recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                        reconstruction[:, [0]], target[:, [0]]
                    )
                    recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                        reconstruction[:, [1]], target[:, [1]]
                    )
                    recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                        reconstruction[:, [2]], target[:, [2]]
                    )
                    recons_loss = recons_loss + perceptual_weight * loss_perceptual(
                        reconstruction[:, [3]], target[:, [3]]
                    )

            if type(recons_loss) == float or recons_loss.isnan().sum() != 0:
                print("NaN found in loss!")
                step_deficit += 1
            else:
                if args.amp:
                    scaler.scale(recons_loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    scaler.step(opt)
                    scaler.update()
                else:
                    recons_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    opt.step()

                epoch_loss += recons_loss.item()

                wandb.log({"train/recon_loss": recons_loss.item()})

            progress_bar.set_postfix(
                {"recon_loss": epoch_loss / (step + 1 - step_deficit)}
            )
        wandb.log({"train/lr": opt.param_groups[0]["lr"]})
        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            inputs = []
            recon_pd = []
            recon_r1 = []
            recon_r2s = []
            recon_mt = []
            val_loss = 0
            val_ssim = 0
            step_deficit = 1e-7
            # val_iter = None
            if args.debug:
                saver1 = mn.transforms.SaveImage(
                    output_dir=os.path.join(args.logdir, args.name, "debug-val"),
                    output_postfix="img",
                    separate_folder=False,
                    print_log=False,
                )
                saver2 = mn.transforms.SaveImage(
                    output_dir=os.path.join(args.logdir, args.name, "debug-val"),
                    output_postfix="pred",
                    separate_folder=False,
                    print_log=False,
                )
                saver3 = mn.transforms.SaveImage(
                    output_dir=os.path.join(args.logdir, args.name, "debug-val"),
                    output_postfix="recon",
                    separate_folder=False,
                    print_log=False,
                )
            n_samples = 4
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    target = batch["mpm"].to(images)
                    params = batch["params"].to(images)
                    mask = batch["mask"].to(device)
                    if args.amp:
                        ctx = torch.autocast(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        scaler = torch.cuda.amp.GradScaler()
                    else:
                        ctx = nullcontext()
                    with ctx:
                        with torch.no_grad():
                            reconstruction = activate(model(images))
                            if val_step < n_samples:
                                recon_ = custom_cc.forward_model(
                                    reconstruction, params, 1
                                )
                                if args.debug:
                                    saver1(torch.Tensor(images[0].cpu().float()))
                                    if args.mask:
                                        saver2(
                                            torch.Tensor(
                                                (reconstruction * mask)[0].cpu().float()
                                            )
                                        )
                                        saver3(
                                            torch.Tensor(
                                                (recon_ * mask)[0].cpu().float()
                                            )
                                        )
                                    else:
                                        saver2(
                                            torch.Tensor(
                                                reconstruction[0].cpu().float()
                                            )
                                        )
                                        saver3(torch.Tensor(recon_[0].cpu().float()))
                            target, reconstruction = normalise_mpm(
                                target, reconstruction, clip=True
                            )
                            recons_loss = crit(reconstruction, target)
                            if args.mask:
                                recons_loss = (
                                    mask * recons_loss
                                )  # mask to only calculate foreground
                                reconstruction = mask * reconstruction
                                target = mask * target
                            recons_loss = recons_loss.mean()
                            if epoch > args.l2_epochs:
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, [0]], target[:, [0]]
                                    )
                                )
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, [1]], target[:, [1]]
                                    )
                                )
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, [2]], target[:, [2]]
                                    )
                                )
                                recons_loss = (
                                    recons_loss
                                    + perceptual_weight
                                    * loss_perceptual(
                                        reconstruction[:, [3]], target[:, [3]]
                                    )
                                )
                            recons_ssim = ssim(reconstruction, target)

                    if type(recons_loss) == float or recons_loss.isnan().sum() != 0:
                        print("NaN found in loss!")
                        step_deficit += 1
                    else:
                        val_loss += recons_loss.item()
                        val_ssim += recons_ssim.item()

                    if val_step < n_samples:
                        inputs.append(
                            images[0, 0, ..., images.size(-1) // 2].cpu().float()[None]
                        )
                        reconstruction = torch.nan_to_num(
                            reconstruction
                        )  # if NaN, just show empty image
                        recon_pd.append(
                            reconstruction[0, 0, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                        recon_r1.append(
                            reconstruction[0, 1, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                        recon_r2s.append(
                            reconstruction[0, 2, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                        recon_mt.append(
                            reconstruction[0, 3, ..., images.size(-1) // 2]
                            .cpu()
                            .float()[None]
                        )
                    elif val_step == n_samples:
                        grid_inputs = make_grid(
                            inputs,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_pd = make_grid(
                            recon_pd,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_r1 = make_grid(
                            recon_r1,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_r2s = make_grid(
                            recon_r2s,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        grid_recon_mt = make_grid(
                            recon_mt,
                            nrow=int(n_samples**0.5),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                        wandb.log(
                            {
                                "val/examples": [
                                    wandb.Image(
                                        grid_inputs[0].numpy(), caption="Input"
                                    ),
                                    wandb.Image(
                                        grid_recon_pd[0].numpy(), caption="Predicted PD"
                                    ),
                                    wandb.Image(
                                        grid_recon_r1[0].numpy(), caption="Predicted R1"
                                    ),
                                    wandb.Image(
                                        grid_recon_r2s[0].numpy(),
                                        caption="Predicted R2s",
                                    ),
                                    wandb.Image(
                                        grid_recon_mt[0].numpy(), caption="Predicted MT"
                                    ),
                                ]
                            }
                        )

                metric = val_ssim / (val_step + 1 - step_deficit)
                wandb.log({"val/recon_loss": val_loss / (val_step + 1 - step_deficit)})
                wandb.log({"val/recon_ssim": val_ssim / (val_step + 1 - step_deficit)})
                print(
                    "Validation complete. Loss: {:.3f} // SSIM: {:.3f}".format(
                        val_loss / (val_step + 1 - step_deficit),
                        val_ssim / (val_step + 1 - step_deficit),
                    )
                )

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
        "--epochs", type=int, default=1000, help="Number of epochs for training."
    )
    parser.add_argument(
        "--epoch_length", type=int, default=200, help="Number of iterations per epoch."
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--val_interval", type=int, default=2, help="Validation interval."
    )
    parser.add_argument(
        "--l2_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs using L2 loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of subjects to use per batch."
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="baur",
        help="Loss function to use. Options: [l1, l2, huber, baur, barron]",
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--mask", default=False, action="store_true")
    parser.add_argument(
        "--logdir", type=str, default="./", help="Path to saved outputs"
    )
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument("--anomaly", default=False, action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. If not specified then will check for CUDA.",
    )
    parser.add_argument(
        "--lowres",
        default=False,
        action="store_true",
        help="Train with un-cropped 2D images.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Save sample images before training.",
    )
    parser.add_argument(
        "--only_stroke",
        default=False,
        action="store_true",
        help="Train only with stroke MPMs.",
    )
    parser.add_argument(
        "--upsample",
        default="transpose",
        type=str,
        help="Method of upsampling. Options: ['transpose', 'subpixel', 'interp'].",
    )
    parser.add_argument(
        '--reset_training',
        action='store_true',
        help='Reset training and treat weights as a pre-trained model. Useful if resuming an existing checkpoint with e.g. a new loss.',
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    print()
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    train_loader, val_loader, train_transform = get_loaders(
        args.batch_size, device, args.lowres, args.only_stroke
    )

    if args.debug:
        saver1 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img",
            separate_folder=False,
            print_log=False,
        )
        saver3 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="mpm",
            separate_folder=False,
            print_log=False,
        )
        for i, batch in enumerate(val_loader):
            if i > 5:
                break
            else:
                print(
                    "Image: ",
                    batch["image"].shape,
                    "min={}".format(batch["image"].min()),
                    "max={}".format(batch["image"].max()),
                )
                saver1(
                    torch.Tensor(batch["image"][0].cpu().float()),
                )
                if "mpm" in batch.keys():
                    print(
                        "MPM: ",
                        batch["mpm"].shape,
                        "min={}".format(batch["mpm"].min()),
                        "max={}".format(batch["mpm"].max()),
                    )
                    saver3(
                        torch.Tensor(batch["mpm"][0].cpu().float()),
                    )

    return args, device, train_loader, val_loader, train_transform


def main():
    args, device, train_loader, val_loader, train_transform = set_up()
    run_model(args, device, train_loader, val_loader, train_transform)


if __name__ == "__main__":
    main()