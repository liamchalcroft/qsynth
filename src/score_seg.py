import monai as mn
import glob
import os
from monai.networks.nets import UNet
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from scipy import ndimage
import warnings
import cc3d


def compute_dice(im1, im2, empty_value=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as im1. If not boolean, it will be converted.
    empty_value : scalar, float.

    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.

    This function has been adapted from the Verse Challenge repository:
    https://github.com/anjany/verse/blob/main/utils/eval_utilities.py
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_value

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


def compute_absolute_volume_difference(im1, im2, voxel_size=1.0):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = float(voxel_size)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


def compute_absolute_lesion_difference(ground_truth, prediction, connectivity=26):
    """
    Computes the absolute lesion difference between two masks. The number of lesions are counted for
    each volume, and their absolute difference is computed.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.

    Returns
    -------
    abs_les_diff : int
        Absolute lesion difference as integer.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    _, ground_truth_numb_lesion = cc3d.connected_components(
        ground_truth, connectivity=connectivity, return_N=True
    )
    _, prediction_numb_lesion = cc3d.connected_components(
        prediction, connectivity=connectivity, return_N=True
    )
    abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)

    return abs_les_diff


def compute_lesion_f1_score(ground_truth, prediction, empty_value=1.0, connectivity=26):
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(ground_truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(
        ground_truth, connectivity=connectivity, return_N=True
    )

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(
            labeled_ground_truth, binary=True, in_place=True
        ):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(
        prediction, connectivity=connectivity, return_N=True
    )
    if N > 0:
        for _, binary_cluster_image in cc3d.each(
            labeled_prediction, binary=True, in_place=True
        ):
            if not np.logical_and(binary_cluster_image, ground_truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        _, N = cc3d.connected_components(
            ground_truth, connectivity=connectivity, return_N=True
        )
        if N == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score


isles_metric = {
    "isles_dice": compute_dice,
    "absolute_volume_difference": compute_absolute_volume_difference,
    "absolute_lesion_difference": compute_absolute_lesion_difference,
    "lesion_f1_score": compute_lesion_f1_score,
}


def run(data, model, device):
    odir = "./seg_scores/"
    save = os.path.join(odir, data, model + "_scores.csv")
    # find files and load with preprocessing etc
    if data == "atlas/t1":
        rawfiles = np.loadtxt("/PATH/TO/ATLAS/atlas_test.txt", dtype=str)
        files = [
            {
                "gt": f.replace("1mm_", ""),
                "pred": "/PATH/TO/SYNTHBLOCH_PREDICTIONS/atlas/t1/"
                + model
                + "/"
                + f.replace("1mm_", "")
                .replace("label-L_desc-T1lesion_mask", "T1w")
                .split("/")[-1]
                .split(".nii")[0]
                + "_pred.nii.gz",
            }
            for f in rawfiles
        ]
    if "isles2015" in data:
        rawfiles = glob.glob(
            "/PATH/TO/SYNTHBLOCH_PREDICTIONS/"
            + data
            + "/"
            + model
            + "/*_pred.nii*"
        )
        files = []
        for f in rawfiles:
            gt_id = glob.glob(
                "/PATH/TO/ISLES/2015/Training/*/*/"
                + f.split("/")[-1].split("_pred")[0]
                + "*.nii*"
            )[0].split("/")[-3]
            gt = glob.glob(
                "/PATH/TO/ISLES/2015/Training/" + gt_id + "/*OT*/*.nii*"
            )
            assert len(gt) == 1
            files.append({"gt": gt[0], "pred": f})
    if "isles2022" in data:
        rawfiles = glob.glob(
            "/PATH/TO/SYNTHBLOCH_PREDICTIONS/"
            + data
            + "/"
            + model
            + "/*_pred.nii*"
        )
        files = []
        for f in rawfiles:
            rootdir = "/PATH/TO/ISLES/2022/dataset-ISLES22^public^unzipped^version/derivatives/"
            froot = f.split("/")[-1].split("_")
            gt = glob.glob(
                os.path.join(
                    rootdir, froot[0], froot[1], froot[0] + "_" + froot[1] + "_msk.nii*"
                )
            )
            assert len(gt) == 1
            files.append({"gt": gt[0], "pred": f})
    if "arc" in data:
        rawfiles = glob.glob(
            "/PATH/TO/SYNTHBLOCH_PREDICTIONS/"
            + data
            + "/"
            + model
            + "/*_pred.nii*"
        )
        files = []
        for f in rawfiles:
            gt_id = glob.glob(
                "/PATH/TO/ARC/PREPROC/*/*/"
                + f.split("/")[-1].split("_pred")[0]
                + "*.nii*"
            )[0]
            gt = glob.glob("/".join(gt_id.split("/")[:-1]) + "/*lesion_mask.nii.gz")
            assert len(gt) == 1
            files.append({"gt": gt[0], "pred": f})
    if "ploras" in data:
        data, seq = data.split("/")
        rawfiles = glob.glob(
            "/PATH/TO/SYNTHBLOCH_PREDICTIONS/"
            + data
            + "/"
            + model
            + "/"
            + seq
            + "/*_pred.nii*"
        )
        files = []
        for f in rawfiles:
            files.append({"gt": f.replace("_pred.nii.gz", "_gt.nii"), "pred": f})

    preproc = mn.transforms.Compose(
        transforms=[
            mn.transforms.LoadImageD(keys=["pred", "gt"], allow_missing_keys=True),
            mn.transforms.EnsureChannelFirstD(
                keys=["pred", "gt"], allow_missing_keys=True
            ),
            mn.transforms.SpacingD(
                keys=["pred", "gt"], pixdim=1, allow_missing_keys=True
            ),
            mn.transforms.ResizeWithPadOrCropD(
                keys=["pred", "gt"],
                spatial_size=(256, 256, 256),
                allow_missing_keys=True,
            ),
            # mn.transforms.SpacingD(keys=["gt"], pixdim=2 if args.lowres else 1, allow_missing_keys=True),
            mn.transforms.ToTensorD(
                keys=["pred", "gt"], device=device, allow_missing_keys=True
            ),
        ]
    )

    os.makedirs("/".join(save.split("/")[:-1]), exist_ok=True)

    all_metrics = []
    for f in tqdm(files, total=len(files)):
        batch = preproc(f)
        pred = torch.nan_to_num(batch["pred"][None])
        trgt = 5 if pred.max() > 1 else 1
        pred = pred == trgt  # needs double checking
        pred = pred.int()
        gt = torch.nan_to_num(batch["gt"][None])

        pred = torch.cat([1.0 - pred.float(), pred.float()], dim=1)
        gt = torch.cat([1.0 - gt.float(), gt.float()], dim=1)

        current = {"File": f["gt"].split("/")[-1]}

        current["dice"] = mn.metrics.compute_dice(
            pred, gt, include_background=False
        ).item()
        current["hd95"] = mn.metrics.compute_hausdorff_distance(
            pred, gt, include_background=False, percentile=95
        ).item()
        cfx = mn.metrics.get_confusion_matrix(pred, gt, include_background=False)
        for mtrc in [
            "sensitivity",
            "specificity",
            "precision",
            "negative predictive value",
            "miss rate",
            "fall out",
            "false discovery rate",
            "false omission rate",
            "prevalence threshold",
            "threat score",
            "accuracy",
            "balanced accuracy",
            "f1 score",
            "matthews correlation coefficient",
            "fowlkes mallows index",
            "informedness",
            "markedness",
        ]:
            current[mtrc] = mn.metrics.compute_confusion_matrix_metric(mtrc, cfx).item()
        for mtrc in [
            "isles_dice",
            "absolute_volume_difference",
            "absolute_lesion_difference",
            "lesion_f1_score",
        ]:
            current[mtrc] = isles_metric[mtrc](
                gt[0, 1].int().cpu().numpy(), pred[0, 1].int().cpu().numpy()
            )
        current["Lesion Volume"] = int(gt[0, 1].sum())
        current["Lesion Count"] = int(ndimage.label(gt[0, 1].int().cpu().numpy())[1])

        all_metrics.append(current)
        df = pd.DataFrame.from_dict(all_metrics)
        df.to_csv(save)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data in [
        # "atlas/t1",
        # "isles2015/t1",
        # "isles2015/t2",
        # "isles2015/flair",
        # "isles2015/dwi",
        # "isles2015/ensemble",
        # 'isles2022/dwi',
        # 'isles2022/flair',
        # 'isles2022/adc',
        # "arc/t1",
        # "arc/t2",
        # "arc/flair",
        # "arc/ensemble",
        "ploras/T2",
        "ploras/FLAIR",
    ]:
        for model in [
            "atlas-real-192-mb",
            "atlas-seg-1mm",
            "bloch-synth-mb-1mm",
            "bloch-mix-mb-1mm",
            "atlas-mix-192-mb",
            "atlas-synth-192-mb",
        ]:
            run(data, model, device)


if __name__ == "__main__":
    main()