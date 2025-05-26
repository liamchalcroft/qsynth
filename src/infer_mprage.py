import os
import shutil
from model import UNet
import torch
import numpy as np
import monai as mn
import argparse
from nitorch.tools.qmri.generators.mprage import mprage
import glob


def activate(x):
    # apply activation function to match postproc in nitorch qmri
    x[:, 0] = x[:, 0].exp()  # pd = e^f(x)
    x[:, 1] = x[:, 1].exp()  # r1 = e^f(x)
    x[:, 2] = x[:, 2].exp()  # r2 = e^f(x)
    x[:, 3] = x[:, 3].neg().exp().add(1).reciprocal().mul(100)  # mt = 100/(1+e^-f(x))
    x[:, 1] = x[:, 1] / 10
    x[:, 2] = x[:, 2] / 10
    return x


def run_model(args, device):
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

    assert os.path.exists(args.weights), "No checkpoints found at path {}.".format(
        args.weights
    )
    checkpoint = torch.load(args.weights, map_location=device)
    print(
        "\nLoading from epoch #{} with WandB ID {}".format(
            checkpoint["epoch"], checkpoint["wandb"]
        )
    )
    model.load_state_dict(checkpoint["net"])
    model.eval()

    files = glob.glob(args.files)
    files_ = []

    os.makedirs(args.savedir, exist_ok=True)
    for i, f in enumerate(files):
        assert os.path.exists(f), "File {} does not exist!".format(f)
        if "_T1w" in f:
            shutil.copyfile(
                f,
                os.path.join(
                    args.savedir, f.split("/")[-1].split("_T1w")[0] + ".nii.gz"
                ),
            )
            files_.append(
                os.path.join(
                    args.savedir, f.split("/")[-1].split("_T1w")[0] + ".nii.gz"
                )
            )
            lab = f.replace("T1w", "label-L_desc-T1lesion_mask")
            shutil.copyfile(lab, os.path.join(args.savedir, lab.split("/")[-1]))
        else:
            shutil.copyfile(
                f, os.path.join(args.savedir, f.split("/")[-1].replace("_reslice", ""))
            )
            files_.append(os.path.join(args.savedir, f.split("/")[-1]))
            lab = f.replace("_reslice", "_lesion")
            shutil.copyfile(lab, os.path.join(args.savedir, lab.split("/")[-1]))

    window = mn.inferers.SlidingWindowInferer(
        3 * [args.patch],
        sw_batch_size=args.batch_size,
        overlap=0.5,
        mode="gaussian",
        sigma_scale=0.125,
        cval=0.0,
        sw_device=None,
        device=None,
        progress=False,
        cache_roi_weight_map=False,
    )

    if args.tta:
        flips = [
            mn.transforms.Flip(spatial_axis=ax)
            for ax in [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        ]

    load = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(keys=["img"]),
            mn.transforms.EnsureChannelFirstD(keys=["img"]),
            mn.transforms.ToTensorD(keys=["img"], device=device),
            mn.transforms.OrientationD(keys=["img"], axcodes="RAS"),
        ]
    )

    get_mask = mn.transforms.Compose(
        transforms=[
            mn.transforms.GaussianSmooth(sigma=11),
            mn.transforms.ForegroundMask(threshold="otsu", invert=True),
            mn.transforms.FillHoles(),
            mn.transforms.Spacing(pixdim=args.pixdim),
        ]
    )

    preproc = mn.transforms.Compose(
        transforms=[
            mn.transforms.SpacingD(keys=["img"], pixdim=args.pixdim),
            mn.transforms.ScaleIntensityRangePercentilesD(
                keys=["img"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                channel_wise=True,
            ),
            mn.transforms.LambdaD(keys=["img"], func=mn.transforms.SignalFillEmpty()),
            mn.transforms.ToTensorD(dtype=torch.float32, keys=["img"]),
        ]
    )

    for f in files_:
        # loop of: infer, post-process, save
        batch = {"img": f.replace("_reslice", "")}
        batch = load(batch)
        mask = get_mask(batch["img"]).to(device)
        batch = preproc(batch)
        img = batch["img"].to(device)
        with torch.no_grad():
            pred = window(img[None], model)[0]
            if args.tta:
                for flip in flips:
                    pred += flip(window(flip(img)[None], model)[0])
                pred /= len(flips)

        # pred.applied_operations = img.applied_operations
        # pred_dict = {}
        # pred_dict["img"] = pred
        # with mn.transforms.utils.allow_missing_keys_mode(preproc):
        #     inverted_pred = preproc.inverse(pred_dict)
        # pred = inverted_pred["img"]
        pred = activate(pred[None])[0]
        # pred = mask * pred

        print("Prediction complete. Saving parameter maps...")
        mn.transforms.SaveImage(
            output_dir=args.savedir,
            output_postfix="PD",
            separate_folder=False,
            print_log=False,
            resample=False,
            dtype=np.float32,
        )(pred[0])
        mn.transforms.SaveImage(
            output_dir=args.savedir,
            output_postfix="R1",
            separate_folder=False,
            print_log=False,
            resample=False,
            dtype=np.float32,
        )(pred[1])
        mn.transforms.SaveImage(
            output_dir=args.savedir,
            output_postfix="R2s",
            separate_folder=False,
            print_log=False,
            resample=False,
            dtype=np.float32,
        )(pred[2])
        mn.transforms.SaveImage(
            output_dir=args.savedir,
            output_postfix="MT",
            separate_folder=False,
            print_log=False,
            resample=False,
            dtype=np.float32,
        )(pred[3])
        mn.transforms.SaveImage(
            output_dir=args.savedir,
            output_postfix="mask",
            separate_folder=False,
            print_log=False,
            resample=False,
            dtype=np.int8,
        )(mask[0])

        print("Parameter maps saved. Predicting simulated MPRAGE from parameters...")
        mprage_ = mprage(pred[0], pred[1], pred[2], device=device)
        mn.transforms.SaveImage(
            output_dir=args.savedir,
            output_postfix="sim_mprage",
            separate_folder=False,
            print_log=False,
            resample=False,
            dtype=np.float32,
        )(mprage_)


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", type=str, help="Path to trained model weights.")
    parser.add_argument("--tta", default=False, action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to use for window inference.",
    )
    parser.add_argument(
        "--patch", type=int, default=128, help="Isotropic patch size for inference."
    )
    parser.add_argument(
        "--pixdim", type=int, default=1, help="Isotropic slice thickness for inference."
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--savedir", type=str, help="Path to save prediction outputs")
    parser.add_argument("--files", default="", help="Glob to desired files.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. If not specified then will check for CUDA.",
    )
    parser.add_argument(
        "--upsample",
        default="transpose",
        type=str,
        help="Method of upsampling. Options: ['transpose', 'subpixel', 'interp'].",
    )
    args = parser.parse_args()

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
    assert args.batch_size == 1, "Currently only support batch size of 1."

    return args, device


def main():
    args, device = set_up()
    run_model(args, device)


if __name__ == "__main__":
    main()