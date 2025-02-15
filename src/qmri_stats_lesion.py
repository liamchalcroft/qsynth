import nibabel as nb
import torch
import cornucopia as cc
import pandas as pd
import os
import glob
import tqdm


pdirs = glob.glob("/PATH/TO/STROKE_QMRI/*/")


df_list = []

for pdir in tqdm.tqdm(pdirs, total=len(pdirs)):
    if os.path.exists(os.path.join(pdir, "lesion_mask.nii")):
        imgs_ = [
            torch.Tensor(nb.load(os.path.join(pdir, "masked_pd.nii")).get_fdata()),
            torch.Tensor(nb.load(os.path.join(pdir, "masked_r1.nii")).get_fdata()),
            torch.Tensor(nb.load(os.path.join(pdir, "masked_r2s.nii")).get_fdata()),
            torch.Tensor(nb.load(os.path.join(pdir, "masked_mt.nii")).get_fdata()),
        ]
        map = torch.Tensor(nb.load(os.path.join(pdir, "lesion_mask.nii")).get_fdata())

        mean = (map * imgs_[0]).sum() / map.sum()
        std = ((map * (imgs_[0] - mean) ** 2).sum() / map.sum()) ** 0.5
        df_list.append(
            {
                "Label": "stroke",
                "Modality": "PD",
                "mu": mean.item(),
                "std": std.item(),
            }
        )

        mean = (map * imgs_[1]).sum() / map.sum()
        std = ((map * (imgs_[1] - mean) ** 2).sum() / map.sum()) ** 0.5
        df_list.append(
            {
                "Label": "stroke",
                "Modality": "R1",
                "mu": mean.item(),
                "std": std.item(),
            }
        )

        mean = (map * imgs_[2]).sum() / map.sum()
        std = ((map * (imgs_[2] - mean) ** 2).sum() / map.sum()) ** 0.5
        df_list.append(
            {
                "Label": "stroke",
                "Modality": "R2s_OLS",
                "mu": mean.item(),
                "std": std.item(),
            }
        )

        mean = (map * imgs_[3]).sum() / map.sum()
        std = ((map * (imgs_[3] - mean) ** 2).sum() / map.sum()) ** 0.5
        df_list.append(
            {
                "Label": "stroke",
                "Modality": "MT",
                "mu": mean.item(),
                "std": std.item(),
            }
        )


df = pd.DataFrame(df_list)

df.to_csv("lesion_stats.csv")