import nibabel as nb
import torch
import cornucopia as cc
import pandas as pd
import os
import glob
import tqdm


pdirs = glob.glob("/PATH/TO/HEALTHY_QMRI/CS*/")


df_list = []

for pdir in tqdm.tqdm(pdirs, total=len(pdirs)):
    if os.path.exists(os.path.join(pdir, "c01_1_00001_sim_mprage_mb.nii")):
        imgs_ = [
            torch.Tensor(nb.load(os.path.join(pdir, "masked_pd.nii")).get_fdata()),
            torch.Tensor(nb.load(os.path.join(pdir, "masked_r1.nii")).get_fdata()),
            torch.Tensor(nb.load(os.path.join(pdir, "masked_r2s.nii")).get_fdata()),
            torch.Tensor(nb.load(os.path.join(pdir, "masked_mt.nii")).get_fdata()),
        ]
        maps = [
            torch.Tensor(
                nb.load(
                    os.path.join(pdir, "c0" + str(i + 1) + "_1_00001_sim_mprage_mb.nii")
                ).get_fdata()
            )
            for i in range(9)
        ]
        bg = torch.ones_like(maps[0]) - torch.stack(maps, -1).sum(-1)
        # imgs_ = [cc.utils.conv.smoothnd(img_, fwhm=[3,3,3]) for img_ in imgs_]

        # df = pd.DataFrame(columns=['Label', 'Modality', 'mu', 'std'])
        # df_list = []

        for i, map in enumerate(maps):
            mean = (map * imgs_[0]).sum() / map.sum()
            std = ((map * (imgs_[0] - mean) ** 2).sum() / map.sum()) ** 0.5
            df_list.append(
                {
                    "Label": "c0" + str(i + 1),
                    "Modality": "PD",
                    "mu": mean.item(),
                    "std": std.item(),
                }
            )

            mean = (map * imgs_[1]).sum() / map.sum()
            std = ((map * (imgs_[1] - mean) ** 2).sum() / map.sum()) ** 0.5
            df_list.append(
                {
                    "Label": "c0" + str(i + 1),
                    "Modality": "R1",
                    "mu": mean.item(),
                    "std": std.item(),
                }
            )

            mean = (map * imgs_[2]).sum() / map.sum()
            std = ((map * (imgs_[2] - mean) ** 2).sum() / map.sum()) ** 0.5
            df_list.append(
                {
                    "Label": "c0" + str(i + 1),
                    "Modality": "R2s_OLS",
                    "mu": mean.item(),
                    "std": std.item(),
                }
            )

            mean = (map * imgs_[3]).sum() / map.sum()
            std = ((map * (imgs_[3] - mean) ** 2).sum() / map.sum()) ** 0.5
            df_list.append(
                {
                    "Label": "c0" + str(i + 1),
                    "Modality": "MT",
                    "mu": mean.item(),
                    "std": std.item(),
                }
            )

        mean = (bg * imgs_[0]).sum() / bg.sum()
        std = ((bg * (imgs_[0] - mean) ** 2).sum() / bg.sum()) ** 0.5
        df_list.append(
            {
                "Label": "background",
                "Modality": "PD",
                "mu": mean.item(),
                "std": std.item(),
            }
        )

        mean = (bg * imgs_[1]).sum() / bg.sum()
        std = ((bg * (imgs_[1] - mean) ** 2).sum() / bg.sum()) ** 0.5
        df_list.append(
            {
                "Label": "background",
                "Modality": "R1",
                "mu": mean.item(),
                "std": std.item(),
            }
        )

        mean = (bg * imgs_[2]).sum() / bg.sum()
        std = ((bg * (imgs_[2] - mean) ** 2).sum() / bg.sum()) ** 0.5
        df_list.append(
            {
                "Label": "background",
                "Modality": "R2s_OLS",
                "mu": mean.item(),
                "std": std.item(),
            }
        )

        mean = (bg * imgs_[3]).sum() / bg.sum()
        std = ((bg * (imgs_[3] - mean) ** 2).sum() / bg.sum()) ** 0.5
        df_list.append(
            {
                "Label": "background",
                "Modality": "MT",
                "mu": mean.item(),
                "std": std.item(),
            }
        )

df = pd.DataFrame(df_list)

df.to_csv("tissue_stats.csv")