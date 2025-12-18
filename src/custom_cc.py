import torch
import cornucopia as cc
import numpy as np
import monai as mn
import typing as tp
from copy import deepcopy
import json
import os
from nitorch.tools import qmri
import pandas as pd
from scipy.stats import median_abs_deviation
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import (
    convert_to_dst_type,
    convert_to_tensor,
)

import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


# df = pd.DataFrame(np.array([
# ['c01','PD',76.83973693847656,8.441889762878418],
# ['c01','R1',0.6519561409950256,0.11394736170768738],
# ['c01','R2s_OLS',19.844810485839844,10.548783302307129],
# ['c01','MT',0.7788642644882202,0.27358347177505493],
# ['c02','PD',75.3653564453125,7.925553321838379],
# ['c02','R1',0.7881048321723938,0.1264655888080597],
# ['c02','R2s_OLS',23.900732040405273,14.468306541442871],
# ['c02','MT',1.0397398471832275,0.32438114285469055],
# ['c03','PD',69.79425048828125,5.409791946411133],
# ['c03','R1',0.9676176309585571,0.08785875886678696],
# ['c03','R2s_OLS',22.22746467590332,7.581535339355469],
# ['c03','MT',1.4996668100357056,0.279540091753006],
# ['c04','PD',74.7887191772461,18.172908782958984],
# ['c04','R1',0.5135526657104492,0.17151211202144623],
# ['c04','R2s_OLS',18.337505340576172,14.0609712600708],
# ['c04','MT',0.3277960419654846,0.3410772681236267],
# ['c05','PD',63.70111846923828,14.998268127441406],
# ['c05','R1',1.340438723564148,0.3041521906852722],
# ['c05','R2s_OLS',67.2598876953125,18.32422637939453],
# ['c05','MT',0.683129072189331,0.7745519280433655],
# ['c06','PD',68.10847473144531,12.654195785522461],
# ['c06','R1',0.9208313822746277,0.21340268850326538],
# ['c06','R2s_OLS',48.71059036254883,16.366018295288086],
# ['c06','MT',0.7272953391075134,0.5496963262557983],
# ['c07','PD',57.921478271484375,18.193452835083008],
# ['c07','R1',1.034616470336914,0.235544353723526],
# ['c07','R2s_OLS',60.5720329284668,19.401432037353516],
# ['c07','MT',0.5690636038780212,0.74372398853302],
# ['c08','PD',49.984378814697266,21.84061050415039],
# ['c08','R1',0.8640786409378052,0.24171693623065948],
# ['c08','R2s_OLS',55.6898193359375,29.8648738861084],
# ['c08','MT',0.3394244909286499,0.4635162353515625],
# ['c09','PD',26.900388717651367,16.071521759033203],
# ['c09','R1',0.8421545624732971,0.20409879088401794],
# ['c09','R2s_OLS',40.46715545654297,28.012388229370117],
# ['c09','MT',0.16868367791175842,0.5573275685310364],
# # ['background','PD',-24.289350509643555,112.48798370361328],
# # ['background','R1',0.2743654251098633,0.40419521927833557],
# # ['background','R2s_OLS',5.470640182495117,23.80196762084961],
# # ['background','MT',0.41837266087532043,1.3029364347457886],
# ['background','PD',0,0],
# ['background','R1',0,0],
# ['background','R2s_OLS',0,0],
# ['background','MT',0,0],
# ]), columns=['Label', 'Modality', 'mu', 'std'],
# )

# df['mu'] = pd.to_numeric(df['mu'])
# df['std'] = pd.to_numeric(df['std'])

# Load tissue statistics CSVs
# These are only needed when use_real_mpms=False (for synthetic GMM generation)
# Try to load them, but don't fail if they're missing
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    df_mb = pd.read_csv(os.path.join(_script_dir, "tissue_stats.csv"))
    df_freesurfer = pd.read_csv(os.path.join(_script_dir, "tissue_stats_freesurfer.csv"))
    df_lesion = pd.read_csv(os.path.join(_script_dir, "lesion_stats.csv"))

    # Only print if in debug/development mode
    if os.getenv("DEBUG_TISSUE_STATS"):
        print("PD")
        print(df_mb[df_mb["Modality"] == "PD"].groupby("Label")["mu"].mean())
        print(df_lesion[df_lesion["Modality"] == "PD"].groupby("Label")["mu"].mean())

        print("R1")
        print(df_mb[df_mb["Modality"] == "R1"].groupby("Label")["mu"].mean())
        print(df_lesion[df_lesion["Modality"] == "R1"].groupby("Label")["mu"].mean())

        print("R2s_OLS")
        print(df_mb[df_mb["Modality"] == "R2s_OLS"].groupby("Label")["mu"].mean())
        print(df_lesion[df_lesion["Modality"] == "R2s_OLS"].groupby("Label")["mu"].mean())

        print("MT")
        print(df_mb[df_mb["Modality"] == "MT"].groupby("Label")["mu"].mean())
        print(df_lesion[df_lesion["Modality"] == "MT"].groupby("Label")["mu"].mean())
except FileNotFoundError as e:
    # CSVs not found - create empty dataframes
    # These will only be used if use_real_mpms=False, which will then fail with a clear error
    warnings.warn(f"Tissue stats CSVs not found: {e}. This is OK if using use_real_mpms=True.")
    df_mb = pd.DataFrame()
    df_freesurfer = pd.DataFrame()
    df_lesion = pd.DataFrame()



param_keys = [
    "te",
    "tr",
    "ti1",
    "ti2",
    "fa1",
    "fa2",
]

seq_keys = ["mprage", "mp2rage", "gre", "fse", "flair", "spgr"]


def reformat_params(sequence, params):
    out_dict = {"field": float(params["receive"].item()) / 10}
    for seq in seq_keys:
        if seq in sequence:
            out_dict[seq] = 1.0
        else:
            out_dict[seq] = 0.0
    for p in param_keys:
        if p in params.keys():
            out_dict[p] = float(params[p])
        else:
            out_dict[p] = 0.0
    if "ti" in params.keys():
        out_dict["ti1"] = float(params["ti"])
    if "fa" in params.keys():
        if isinstance(params["fa"], (tuple, list)):
            out_dict["fa1"] = float(params["fa"][0])
            out_dict["fa2"] = float(params["fa"][1])
        else:
            out_dict["fa1"] = float(params["fa"])
    # add rescaling to [0,1] to make hypernetworks work better
    out_dict["te"] = out_dict["te"] / 1000
    out_dict["tr"] = out_dict["tr"] / 1000
    out_dict["ti1"] = out_dict["ti1"] / 1000
    out_dict["ti2"] = out_dict["ti2"] / 1000
    out_dict["fa1"] = out_dict["fa1"] / 180
    out_dict["fa2"] = out_dict["fa2"] / 180
    return out_dict


dict_keys = [
    "field",
    "mprage",
    "mp2rage",
    "gre",
    "fse",
    "flair",
    "spgr",
    "te",
    "tr",
    "ti1",
    "ti2",
    "fa1",
    "fa2",
]


def forward_model(mpm, params, num_ch=1):
    params = torch.chunk(
        params[0].detach().cpu(), num_ch, dim=-1
    )  # assume batch size 1
    outputs = []
    for p in params:
        if p.sum() > 0:
            out_dict = {k: v for k, v in zip(dict_keys, p.tolist())}
            # print(f"out_dict: {out_dict}")
            out_dict["field"] = out_dict["field"] * 10
            out_dict["te"] = out_dict["te"] * 1000
            out_dict["tr"] = out_dict["tr"] * 1000
            out_dict["ti1"] = out_dict["ti1"] * 1000
            out_dict["ti2"] = out_dict["ti2"] * 1000
            out_dict["fa1"] = out_dict["fa1"] * 180
            out_dict["fa2"] = out_dict["fa2"] * 180
            # print(f"out_dict: {out_dict}")
            if out_dict["mprage"] == 1:
                out = qmri.generators.mprage(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    ti=out_dict["ti1"],
                    te=out_dict["te"],
                    fa=out_dict["fa1"],
                    device=mpm.device,
                )
            elif out_dict["mp2rage"] == 1:
                out = qmri.generators.mp2rage(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    ti1=out_dict["ti1"],
                    ti2=out_dict["ti2"],
                    te=out_dict["te"],
                    fa=(out_dict["fa1"], out_dict["fa2"]),
                    device=mpm.device,
                )
            elif out_dict["gre"] == 1:
                out = qmri.gre(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    mt=mpm[0, 3],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    te=out_dict["te"],
                    fa=out_dict["fa1"],
                    device=mpm.device,
                ).volume
            elif out_dict["fse"] == 1:
                out = qmri.generators.fse(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    te=out_dict["te"],
                    device=mpm.device,
                )
            elif out_dict["flair"] == 1:
                out = qmri.generators.flair(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2=mpm[0, 2],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    ti=out_dict["ti1"],
                    te=out_dict["te"],
                    device=mpm.device,
                )
            elif out_dict["spgr"] == 1:
                out = qmri.generators.spgr(
                    pd=mpm[0, 0],
                    r1=mpm[0, 1],
                    r2s=mpm[0, 2],
                    mt=mpm[0, 3],
                    receive=torch.Tensor([out_dict["field"]])[None][None],
                    tr=out_dict["tr"],
                    te=out_dict["te"],
                    fa=out_dict["fa1"],
                    device=mpm.device,
                )
            if len(out.shape) == 4:
                out = out[0]
            outputs.append(out)
        else:
            outputs.append(torch.zeros_like(mpm[0, 0]))
    return torch.stack(outputs, dim=0)[None]


def ensure_list(x):
    if type(x) != list:
        if type(x) == tuple:
            return list(x)
        else:
            return [x]
    else:
        return x


class log10norm:
    def __init__(self, mu, std=1):
        self.mu = np.log10(mu)
        self.std = std

    def __call__(self):
        return 10 ** np.random.normal(self.mu, self.std)


class uniform:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self):
        return np.random.uniform(self.low, self.high)


class log10uniform:
    def __init__(self, low, high):
        self.low = np.log10(low)
        self.high = np.log10(high)

    def __call__(self):
        return 10 ** np.random.uniform(self.low, self.high)


class QMRITransform(cc.Transform):
    def __init__(
        self,
        flair_params={
            "te": log10norm(20e-3, 0.1),
            "tr": log10uniform(1e-3, 5000e-3),
            "ti": log10uniform(1e-3, 3000e-3),
        },
        fse_params={
            "te": log10uniform(1e-3, 3000e-3),
            "tr": log10uniform(1e-3, 3000e-3),
        },
        mp2rage_params={
            "tr": 2300e-3,
            "ti1": uniform(600e-3, 900e-3),
            "ti2": 2200e-3,
            "tx": log10norm(5.8e-3, 0.5),
            "te": log10norm(2.9e-3, 0.5),
            "fa": (uniform(3, 6), uniform(3, 6)),
            "n": 160,
            "eff": 0.96,
        },
        mprage_params={
            "tr": 2300e-3,
            "ti": uniform(600e-3, 900e-3),
            "tx": uniform(4e-3, 8e-3),
            "te": uniform(2e-3, 4e-3),
            "fa": uniform(5, 12),
            "n": 160,
            "eff": 0.96,
        },
        mprage_t1_params={
            "tr": uniform(1900e-3, 2500e-3),
            "ti": uniform(600e-3, 1200e-3),
            "te": uniform(2e-3, 4e-3),
            "fa": uniform(5, 12),
            "n": uniform(100, 200),
            "eff": uniform(0.8, 1.0),
        },
        spgr_params={
            "te": log10uniform(2e-3, 80e-3),
            "tr": log10uniform(5e-3, 800e-3),
            "fa": uniform(5, 50),
        },
        gre_params={
            "te": log10uniform(2e-3, 80e-3),
            "tr": log10uniform(5e-3, 5000e-3),
            "fa": uniform(5, 50),
        },
        sequence=["mprage", "mp2rage", "gre", "fse", "flair", "spgr"],
        field_strength=(0.3, 7),
    ):
        super().__init__()
        self.params = {
            "mprage": mprage_params,
            "mp2rage": mp2rage_params,
            "mprage-t1": mprage_t1_params,
            "gre": gre_params,
            "fse": fse_params,
            "flair": flair_params,
            "spgr": spgr_params,
        }
        self.funcs = {
            "mprage": qmri.generators.mprage,
            "mp2rage": qmri.generators.mp2rage,
            "mprage-t1": qmri.generators.mprage,
            "gre": qmri.gre,
            "fse": qmri.generators.fse,
            "flair": qmri.generators.flair,
            "spgr": qmri.generators.spgr,
        }
        self.sequence = ensure_list(sequence)
        self.field_strength = field_strength

    def sample(self, param, key, sequence):
        if isinstance(param, (float, int)):
            param = np.random.randn() * (param * 0.1) + param
            if param < 0:
                param = -param
            if key == "n":
                param = int(param)
            if key == "eff" and param > 1:
                param = 2 - param
        elif isinstance(param, (log10norm, log10uniform, uniform)):
            param = param()
        elif param == "loguni":
            param = 10 ** (np.random.uniform(-3, 3))
        elif param == "uni":
            param = np.random.uniform(0, 90 if sequence not in ["mprage"] else 10)
        return param

    def get_parameters(self):
        parameters = dict()
        parameters["sequence"] = self.sequence[np.random.randint(len(self.sequence))]
        params = deepcopy(self.params[parameters["sequence"]])
        for key in params.keys():
            if isinstance(params[key], (list, tuple)):
                params[key] = [
                    self.sample(val, key, parameters["sequence"]) for val in params[key]
                ]
            else:
                params[key] = self.sample(params[key], key, parameters["sequence"])
        params["receive"] = torch.Tensor([np.random.uniform(*self.field_strength)])[
            None
        ][None]
        parameters["params"] = params
        parameters["func"] = self.funcs[parameters["sequence"]]
        self.parameters = parameters

        return parameters

    def forward(self, pd, r1, r2s, mt):
        theta = self.get_parameters()
        return self.apply_transform(pd, r1, r2s, mt, theta)

    def apply_transform(self, pd, r1, r2s, mt, parameters):
        in_ = (
            [pd, r1, r2s]
            if parameters["sequence"] in ["fse", "flair", "mprage", "mp2rage"]
            else [pd, r1, r2s, mt]
        )
        out_ = parameters["func"](*in_, **parameters["params"])
        return out_.volume[0] if parameters["sequence"] == "gre" else out_


class RandomGaussianMixtureTransform(torch.nn.Module):
    def __init__(
        self,
        mu=255,
        sigma=16,
        fwhm=2,
        background=None,
        mu_sigma=None,
        sigma_sigma=None,
        dtype=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.sample = dict(
            mu=cc.random.Normal(mu, mu_sigma or [mu_ / 10 for mu_ in mu]),
            sigma=cc.random.Normal(
                sigma, sigma_sigma or [sigma_ / 10 for sigma_ in sigma]
            ),
            fwhm=cc.random.Uniform.make(cc.random.make_range(0, fwhm)),
        )
        self.background = background

    def forward(self, x):
        theta = self.get_parameters(x)
        return self.apply_transform(x, theta)

    def get_parameters(self, x):
        mu = self.sample["mu"]()
        sigma = self.sample["sigma"]()
        fwhm = int(self.sample["fwhm"]())
        if x.dtype.is_floating_point:
            backend = dict(dtype=x.dtype, device=x.device)
        else:
            backend = dict(
                dtype=self.dtype or torch.get_default_dtype(), device=x.device
            )
        mu = torch.as_tensor(mu).to(**backend)
        sigma = torch.as_tensor(sigma).to(**backend)
        return mu, sigma, fwhm

    def apply_transform(self, x, parameters):
        mu, sigma, fwhm = parameters

        backend = dict(dtype=x.dtype, device=x.device)

        mu = torch.nan_to_num(mu.to(**backend))
        sigma = torch.nan_to_num(sigma.to(**backend))
        # Limit mu/sigma to the first N labels present in x
        mu = mu[: x.size(0)]
        sigma = sigma[: x.size(0)]
        if x.size(0) == 1 and x.max() > 1:  # int labels
            x = x.int()
            y = torch.zeros_like(x, **backend)
            nk = len(mu)
            for k in range(nk):
                muk, sigmak = mu[k], sigma[k]
                if self.background is not None and k == self.background:
                    continue
                mask = (x == k).to(**backend)
                # mask = cc.utils.conv.smoothnd(mask, fwhm=[3]*3)
                y1 = torch.randn(x.shape, **backend)
                y1 = cc.utils.conv.smoothnd(y1, fwhm=[fwhm] * 3)
                # y += y1.mul_(sigmak).add_(muk).masked_fill_(x != k, 0)
                y += y1.mul_(sigmak).add_(muk).mul_(mask).abs()
        else:
            y = torch.zeros_like(x[0], **backend)
            if self.background is not None:
                x[self.background] = 0
            y1 = torch.randn(*x.shape, **backend)
            fwhm = [fwhm] * 3
            y1 = cc.utils.conv.smoothnd(y1, fwhm=fwhm)
            y1 = y1.mul_(sigma[..., None, None, None]).add_(mu[..., None, None, None])
            y = torch.sum(x * torch.abs(y1), dim=0)
            y = y[None]
        return y


def donothing(x):
    return x


class QMRISynthFromLabelTransform(cc.Transform):
    def __init__(
        self,
        num_ch=1,
        patch=None,
        rotation=15,
        shears=0.012,
        zooms=0.15,
        elastic=0.05,
        elastic_nodes=10,
        elastic_steps=0,
        gmm_fwhm=10,
        bias=7,
        gamma=False,
        motion_fwhm=3,
        resolution=8,
        snr=10,
        gfactor=5,
        order=3,
        sequence=["mprage", "mp2rage", "gre", "fse", "flair", "spgr"],
        field_strength=(0.3, 7),
        sigma_temp=0.1,
        label_source="mb",
        no_augs=False,
        use_real_mpms=False,
    ):
        super().__init__()
        if label_source == "mb":
            df = df_mb
        elif label_source == "freesurfer":
            df = df_freesurfer
            sigma_temp /= 10
        self.no_augs = no_augs
        self.deform = (
            donothing
            if no_augs
            else cc.RandomAffineElasticTransform(
                elastic,
                elastic_nodes,
                order=order,
                bound="zeros",
                steps=elastic_steps,
                rotations=rotation,
                shears=shears,
                zooms=zooms,
                patch=patch,
            )
        )
        self.use_real_mpms = use_real_mpms
        if not use_real_mpms:
            self.gmm_pd = RandomGaussianMixtureTransform(
                mu=df[df["Modality"] == "PD"].groupby("Label")["mu"].mean().to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "PD"]
                    .groupby("Label")["mu"]
                    .mean()
                ],
                mu_sigma=df[df["Modality"] == "PD"]
                .groupby("Label")["mu"]
                .apply(median_abs_deviation)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "PD"]
                    .groupby("Label")["mu"]
                    .apply(median_abs_deviation)
                ],
                sigma=df[df["Modality"] == "PD"]
                .groupby("Label")["std"]
                .mean()
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "PD"]
                    .groupby("Label")["std"]
                    .mean()
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                sigma_sigma=df[df["Modality"] == "PD"]
                .groupby("Label")["std"]
                .apply(median_abs_deviation)
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "PD"]
                    .groupby("Label")["std"]
                    .apply(median_abs_deviation)
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                fwhm=gmm_fwhm,
                background=0,
            )
            self.gmm_r1 = RandomGaussianMixtureTransform(
                mu=df[df["Modality"] == "R1"].groupby("Label")["mu"].mean().to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R1"]
                    .groupby("Label")["mu"]
                    .mean()
                ],
                mu_sigma=df[df["Modality"] == "R1"]
                .groupby("Label")["mu"]
                .apply(median_abs_deviation)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R1"]
                    .groupby("Label")["mu"]
                    .apply(median_abs_deviation)
                ],
                sigma=df[df["Modality"] == "R1"]
                .groupby("Label")["std"]
                .mean()
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R1"]
                    .groupby("Label")["std"]
                    .mean()
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                sigma_sigma=df[df["Modality"] == "R1"]
                .groupby("Label")["std"]
                .apply(median_abs_deviation)
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R1"]
                    .groupby("Label")["std"]
                    .apply(median_abs_deviation)
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                fwhm=gmm_fwhm,
                background=0,
            )
            self.gmm_r2s = RandomGaussianMixtureTransform(
                mu=df[df["Modality"] == "R2s_OLS"]
                .groupby("Label")["mu"]
                .mean()
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R2s_OLS"]
                    .groupby("Label")["mu"]
                    .mean()
                ],
                mu_sigma=df[df["Modality"] == "R2s_OLS"]
                .groupby("Label")["mu"]
                .apply(median_abs_deviation)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R2s_OLS"]
                    .groupby("Label")["mu"]
                    .apply(median_abs_deviation)
                ],
                sigma=df[df["Modality"] == "R2s_OLS"]
                .groupby("Label")["std"]
                .mean()
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R2s_OLS"]
                    .groupby("Label")["std"]
                    .mean()
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                sigma_sigma=df[df["Modality"] == "R2s_OLS"]
                .groupby("Label")["std"]
                .apply(median_abs_deviation)
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "R2s_OLS"]
                    .groupby("Label")["std"]
                    .apply(median_abs_deviation)
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                fwhm=gmm_fwhm,
                background=0,
            )
            self.gmm_mt = RandomGaussianMixtureTransform(
                mu=df[df["Modality"] == "MT"].groupby("Label")["mu"].mean().to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "MT"]
                    .groupby("Label")["mu"]
                    .mean()
                ],
                mu_sigma=df[df["Modality"] == "MT"]
                .groupby("Label")["mu"]
                .apply(median_abs_deviation)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "MT"]
                    .groupby("Label")["mu"]
                    .apply(median_abs_deviation)
                ],
                sigma=df[df["Modality"] == "MT"]
                .groupby("Label")["std"]
                .mean()
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "MT"]
                    .groupby("Label")["std"]
                    .mean()
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                sigma_sigma=df[df["Modality"] == "MT"]
                .groupby("Label")["std"]
                .apply(median_abs_deviation)
                .apply(lambda x: x * sigma_temp)
                .to_list()
                + [
                    df_lesion[df_lesion["Modality"] == "MT"]
                    .groupby("Label")["std"]
                    .apply(median_abs_deviation)
                    .apply(lambda x: x * sigma_temp)
                ],  # temperature rescaling for sigma
                fwhm=gmm_fwhm,
                background=0,
            )
        self.qmri = QMRITransform(sequence=sequence, field_strength=field_strength)
        self.intensity = (
            donothing
            if no_augs
            else cc.IntensityTransform(
                bias, gamma, motion_fwhm, resolution, snr, gfactor, order
            )
        )
        self.num_ch = num_ch
        self.quantile = donothing if no_augs else cc.QuantileTransform(pmin=0.05, pmax=0.95, clip=True)
        self.smooth = donothing if no_augs else cc.SmoothTransform(fwhm=1)

    def get_parameters(self, x):
        parameters = dict()
        if not self.use_real_mpms:
            parameters["gmm_pd"] = self.gmm_pd.get_parameters(x)
            parameters["gmm_r1"] = self.gmm_r1.get_parameters(x)
            parameters["gmm_r2s"] = self.gmm_r2s.get_parameters(x)
            parameters["gmm_mt"] = self.gmm_mt.get_parameters(x)
        parameters["qmri"] = [self.qmri.get_parameters() for i in range(self.num_ch)]
        if not self.no_augs:
            parameters["deform"] = self.deform.get_parameters(x)
        return parameters

    def forward(self, x):
        theta = self.get_parameters(x)
        self.theta = theta
        return self.apply_transform(x, theta)

    def apply_transform(self, lab, parameters=None):
        # if not self.no_augs:
        #     lab = self.deform.apply_transform(lab, parameters['deform'])
        if not self.use_real_mpms:
            pd = self.gmm_pd.apply_transform(lab, parameters["gmm_pd"])
            r1 = self.gmm_r1.apply_transform(lab, parameters["gmm_r1"])
            r2s = self.gmm_r2s.apply_transform(lab, parameters["gmm_r2s"])
            mt = self.gmm_mt.apply_transform(lab, parameters["gmm_mt"])
        else:
            pd, r1, r2s, mt = torch.chunk(lab, 4, dim=0)
        if not self.no_augs:
            lab = self.deform.apply_transform(lab, parameters["deform"])
            pd, r1, r2s, mt = [
                self.deform.apply_transform(m, parameters["deform"])
                for m in [pd, r1, r2s, mt]
            ]
        # img = [self.smooth(self.quantile(torch.nan_to_num(self.qmri.apply_transform(pd, r1, r2s, mt, parameters['qmri'][i])))) * lab.sum(0) for i in range(self.num_ch)]
        img = []
        for i in range(self.num_ch):
            transformed = torch.nan_to_num(
                self.qmri.apply_transform(pd, r1, r2s, mt, parameters["qmri"][i])
            )
            if torch.all(transformed == 0):
                print("WARNING: The output of nan_to_num is all zeros.")
                img.append(transformed)
            else:
                img.append(self.smooth(self.quantile(transformed)))
        output = [
            torch.cat([self.intensity(img[i]) for i in range(self.num_ch)], dim=0)
        ]
        output.append(lab)
        output.append(torch.cat([pd, r1, r2s, mt], dim=0))
        params = [
            reformat_params(p["sequence"], p["params"]) for p in parameters["qmri"]
        ]
        output.append(
            torch.cat([torch.tensor(list(p.values())) for p in params], dim=0).to(
                lab.device
            )
        )
        return tuple(output)


class SynthBloch:
    def __init__(
        self,
        label_key,
        image_key="image",
        mpm_key="mpm",
        coreg_keys=None,
        num_ch=1,
        patch=None,  # currently not working how we want - generates random crops for all coregs rather than uniform crop
        rotation=15,
        shears=0.012,
        zooms=0.15,
        elastic=0.05,
        elastic_nodes=10,
        gmm_fwhm=10,
        bias=7,
        gamma=0.6,
        motion_fwhm=3,
        resolution=8,
        snr=10,
        gfactor=5,
        order=3,
        label_source="mb",
        no_augs=False,
        use_real_mpms=False,
        sequence=["mprage", "mp2rage", "gre", "fse", "flair", "spgr"],
    ) -> None:
        self.label_key = label_key
        self.image_key = image_key
        self.mpm_key = mpm_key
        self.coreg_keys = coreg_keys
        self.use_real_mpms = use_real_mpms
        self.transform = QMRISynthFromLabelTransform(
            num_ch=num_ch,
            patch=patch,
            rotation=rotation,
            shears=shears,
            zooms=zooms,
            elastic=elastic,
            elastic_nodes=elastic_nodes,
            gmm_fwhm=gmm_fwhm,
            bias=bias,
            gamma=gamma,
            motion_fwhm=motion_fwhm,
            resolution=resolution,
            snr=snr,
            gfactor=gfactor,
            order=order,
            label_source=label_source,
            no_augs=no_augs,
            use_real_mpms=use_real_mpms,
            sequence=sequence,
        )

    def __call__(self, data):
        d = dict(data)
        img, lab, mpm, params = self.transform(d[self.label_key])
        if self.coreg_keys is not None:
            for c in self.coreg_keys:
                if not self.transform.no_augs:
                    d[c] = self.transform.deform.apply_transform(
                        d[c], self.transform.theta["deform"]
                    )
        d[self.image_key] = img
        d[self.label_key] = lab
        d[self.mpm_key] = mpm
        d["params"] = params
        if self.label_key + "_meta_dict" in list(d.keys()):
            d[self.image_key + "_meta_dict"] = d[self.label_key + "_meta_dict"]
            d[self.mpm_key + "_meta_dict"] = d[self.label_key + "_meta_dict"]
        return d


class RandomSkullStrip(mn.transforms.MapTransform, mn.transforms.Randomizable):
    """ """

    backend = [
        mn.utils.enums.TransformBackends.TORCH,
        mn.utils.enums.TransformBackends.NUMPY,
    ]

    def __init__(
        self,
        label_key="label",
        image_key="image",
        out_key="mask",
        dilate_prob=0.3,
        erode_prob=0.3,
    ) -> None:
        mn.transforms.MapTransform.__init__(self, [label_key], allow_missing_keys=False)
        self.label_key = label_key
        self.image_keys = (
            [image_key] if not isinstance(image_key, (list, tuple)) else image_key
        )
        self.out_key = out_key
        # self.fill = mn.transforms.FillHoles()
        # self.dilate = cc.DilateLabelTransform(radius=2)
        # self.r_dilate = cc.RandomDilateLabelTransform(labels=dilate_prob, radius=2)
        # self.r_erode = cc.RandomErodeLabelTransform(labels=erode_prob, radius=4)

    def __call__(
        self, data: tp.Mapping[tp.Hashable, mn.config.type_definitions.NdarrayOrTensor]
    ) -> tp.Dict[tp.Hashable, mn.config.type_definitions.NdarrayOrTensor]:
        d = dict(data)
        mask = (d[self.label_key] > 0.5).int()
        # for i in range(3):
        #     mask = self.dilate(mask)
        #     mask = self.fill(mask)
        # mask = self.r_dilate(mask)
        # mask = self.r_erode(mask)
        for image_key in self.image_keys:
            d[image_key] = mask * d[image_key]
        d[self.label_key] = mask * d[self.label_key]
        del mask
        return d


class LoadMetadataD(mn.transforms.MapTransform):
    def __init__(
        self,
        keys,
        filter=False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.filter = filter

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.filter:
                try:
                    with open(d[key]) as file:
                        json_dict = json.load(file)
                    out_dict = {
                        "field": float(json_dict["MagneticFieldStrength"]),
                        "mprage": "MPRAGE" in json_dict["ScanningSequence"],
                        "mp2rage": "MP2RAGE" in json_dict["ScanningSequence"],
                        "gre": "GR" in json_dict["ScanningSequence"],
                        "fse": "SE" in json_dict["ScanningSequence"],
                        "flair": "IR" in json_dict["ScanningSequence"],
                        "spgr": "SPGR" in json_dict["ScanningSequence"],
                        "te": float(
                            json_dict["EchoTime"]
                            if "EchoTime" in json_dict.keys()
                            else 0.0
                        ),
                        "tr": float(
                            json_dict["RepetitionTime"]
                            if "RepetitionTime" in json_dict.keys()
                            else 0.0
                        ),
                        "ti1": (
                            float(json_dict["InversionTime"])
                            if "InversionTime" in json_dict.keys()
                            else 0.0
                        ),
                        "ti2": float(
                            json_dict["InversionTime2"]
                            if "InversionTime2" in json_dict.keys()
                            else 0.0
                        ),
                        "fa1": float(
                            json_dict["FlipAngle"]
                            if "FlipAngle" in json_dict.keys()
                            else 0.0
                        ),
                        "fa2": float(
                            json_dict["FlipAngle2"]
                            if "FlipAngle2" in json_dict.keys()
                            else 0.0
                        ),
                    }
                    if (
                        sum(
                            out_dict[k]
                            for k in [
                                "mprage",
                                "mp2rage",
                                "gre",
                                "fse",
                                "flair",
                                "spgr",
                            ]
                        )
                        == 1
                    ):
                        # add rescaling to [0,1] to make hypernetworks work better
                        out_dict["field"] = out_dict["field"] / 10
                        out_dict["te"] = out_dict["te"] / 1000
                        out_dict["tr"] = out_dict["tr"] / 1000
                        out_dict["ti1"] = out_dict["ti1"] / 1000
                        out_dict["ti2"] = out_dict["ti2"] / 1000
                        out_dict["fa1"] = out_dict["fa1"] / 180
                        out_dict["fa2"] = out_dict["fa2"] / 180
                        d[key] = torch.tensor(list(out_dict.values()))
                    else:
                        d[key] = d[key]
                except:
                    d[key] = d[key]
            else:
                path = d[key] if isinstance(d[key], (list, tuple)) else [d[key]]
                output = []
                for path_ in path:
                    print(path_)
                    with open(path_) as file:
                        json_dict = json.load(file)
                    out_dict = {
                        "field": float(json_dict["MagneticFieldStrength"]),
                        "mprage": "MPRAGE" in json_dict["ScanningSequence"],
                        "mp2rage": "MP2RAGE" in json_dict["ScanningSequence"],
                        "gre": "GR" in json_dict["ScanningSequence"],
                        "fse": "SE" in json_dict["ScanningSequence"],
                        "flair": "IR" in json_dict["ScanningSequence"],
                        "spgr": "SPGR" in json_dict["ScanningSequence"],
                        "te": float(
                            json_dict["EchoTime"]
                            if "EchoTime" in json_dict.keys()
                            else 0.0
                        ),
                        "tr": float(
                            json_dict["RepetitionTime"]
                            if "RepetitionTime" in json_dict.keys()
                            else 0.0
                        ),
                        "ti1": (
                            float(json_dict["InversionTime"])
                            if "InversionTime" in json_dict.keys()
                            else 0.0
                        ),
                        "ti2": float(
                            json_dict["InversionTime2"]
                            if "InversionTime2" in json_dict.keys()
                            else 0.0
                        ),
                        "fa1": float(
                            json_dict["FlipAngle"]
                            if "FlipAngle" in json_dict.keys()
                            else 0.0
                        ),
                        "fa2": float(
                            json_dict["FlipAngle2"]
                            if "FlipAngle2" in json_dict.keys()
                            else 0.0
                        ),
                    }
                    # add rescaling to [0,1] to make hypernetworks work better
                    out_dict["field"] = out_dict["field"] / 10
                    out_dict["te"] = out_dict["te"] / 1000
                    out_dict["tr"] = out_dict["tr"] / 1000
                    out_dict["ti1"] = out_dict["ti1"] / 1000
                    out_dict["ti2"] = out_dict["ti2"] / 1000
                    out_dict["fa1"] = out_dict["fa1"] / 180
                    out_dict["fa2"] = out_dict["fa2"] / 180
                    output.append(torch.tensor(list(out_dict.values())))
                d[key] = torch.cat(output, dim=0)
        return d


class ClipPercentilesD(mn.transforms.MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRangePercentiles`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower percentile.
        upper: upper percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max]
        channel_wise: if True, compute intensity percentile and normalize every channel separately.
            default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = mn.transforms.ScaleIntensityRangePercentiles.backend

    def __init__(
        self,
        keys,
        lower: float,
        upper: float,
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.lower = lower
        self.upper = upper
        self.channel_wise = channel_wise

    def _normalize(self, img):
        a_min = percentile(img, self.lower)
        a_max = percentile(img, self.upper)
        img = clip(img, a_min, a_max)
        img = convert_to_tensor(img, track_meta=False)
        return img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            img = convert_to_tensor(img, track_meta=get_track_meta())
            img_t = convert_to_tensor(img, track_meta=False)
            if self.channel_wise:
                img_t = torch.stack([self._normalize(img=d) for d in img_t])  # type: ignore
            else:
                img_t = self._normalize(img=img_t)
            d[key] = convert_to_dst_type(img_t, dst=img)[0]
        return d


class ZeroBackgroundD(mn.transforms.MapTransform):
    """
    Zero out background voxels using a mask.

    This is important for BIDS data which may have non-zero values outside
    the brain mask, which can cause training instability when not using --mask flag.

    Args:
        keys: keys of the data to zero out (e.g., ["label"])
        mask_key: key containing the mask (default: "mask")
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys,
        mask_key: str = "mask",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)

        if self.mask_key not in d:
            return d

        mask = d[self.mask_key]
        mask = convert_to_tensor(mask, track_meta=get_track_meta())

        for key in self.key_iterator(d):
            img = d[key]
            img = convert_to_tensor(img, track_meta=get_track_meta())

            # Zero out background (where mask == 0)
            # Expand mask to match image channels if needed
            if mask.ndim == 3 and img.ndim == 4:
                # mask is (H,W,D), img is (C,H,W,D)
                mask_expanded = mask.unsqueeze(0)  # (1,H,W,D)
            elif mask.ndim == 4 and mask.shape[0] == 1:
                # mask is already (1,H,W,D)
                mask_expanded = mask
            else:
                mask_expanded = mask

            # Apply mask
            img_masked = img * (mask_expanded > 0)

            d[key] = convert_to_dst_type(img_masked, dst=img)[0]

        return d


class ScaleBIDSqMRID(mn.transforms.MapTransform):
    """
    Scale BIDS qMRI data to match HEALTHY/STROKE data units AND clip to match distribution.

    BIDS data uses different units/scaling:
    - PD: needs to be scaled by 100
    - R1: needs to be scaled by 1/1000 (convert from ms^-1 to s^-1), then clipped to match H/S range
    - R2*: clip negative values
    - MT: clip negative values

    The clipping is needed because BIDS has a tighter distribution than HEALTHY/STROKE,
    and the model was trained on the noisier HEALTHY/STROKE distribution.

    Args:
        keys: keys of the corresponding items to be transformed (e.g., ["label"])
        path_key: key containing the file path to detect BIDS data (default: "path")
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys,
        path_key: str = "path",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.path_key = path_key

    def _is_bids_data(self, data):
        """Check if data comes from BIDS dataset based on file path"""
        if self.path_key not in data:
            return False

        path = data[self.path_key]
        # Handle different path formats
        if isinstance(path, (list, tuple)):
            path = path[0] if len(path) > 0 else ""
        path_str = str(path)

        # Check if path contains BIDS directory structure
        return "/BIDS/" in path_str or "\\BIDS\\" in path_str

    def __call__(self, data):
        d = dict(data)

        # Only apply scaling if this is BIDS data
        if not self._is_bids_data(d):
            return d

        for key in self.key_iterator(d):
            img = d[key]
            img = convert_to_tensor(img, track_meta=get_track_meta())

            # Apply channel-specific scaling
            # Assuming channel order: [PD, R1, R2*, MT]
            if img.shape[0] >= 4:  # Ensure we have at least 4 channels
                img_scaled = img.clone()
                img_scaled[0] = img[0] * 100.0      # PD: scale by 100
                img_scaled[1] = img[1] / 1000.0     # R1: convert ms^-1 to s^-1

                # Clip R2* and MT to remove negative values (unphysical)
                img_scaled[2] = torch.clamp(img[2], min=0)  # R2*: clip negative values
                img_scaled[3] = torch.clamp(img[3], min=0)  # MT: clip negative values

                d[key] = convert_to_dst_type(img_scaled, dst=img)[0]
            else:
                # If unexpected number of channels, don't scale
                d[key] = img

        return d