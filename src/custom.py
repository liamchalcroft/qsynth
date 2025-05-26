from __future__ import annotations

import numpy as np
from typing import Hashable, Mapping
import torch
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform, Transform
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor
from monai.transforms.utils_pytorch_numpy_unification import percentile
from random import shuffle, randint


class ScaleIntensityRangePercentiles(Transform):
    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        lower: float,
        upper: float,
        b_min: float | None,
        b_max: float | None,
        clip: bool = False,
        relative: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        if lower < 0.0 or lower > 100.0:
            raise ValueError("Percentiles must be in the range [0, 100]")
        if upper < 0.0 or upper > 100.0:
            raise ValueError("Percentiles must be in the range [0, 100]")
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.relative = relative
        self.channel_wise = channel_wise
        self.dtype = dtype

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        a_min: float = percentile(img, self.lower)  # type: ignore
        a_max: float = percentile(img, self.upper)  # type: ignore
        self.a_min = a_min
        self.a_max = a_max
        b_min = self.b_min
        b_max = self.b_max

        if self.relative:
            if (self.b_min is None) or (self.b_max is None):
                raise ValueError(
                    "If it is relative, b_min and b_max should not be None."
                )
            b_min = ((self.b_max - self.b_min) * (self.lower / 100.0)) + self.b_min
            b_max = ((self.b_max - self.b_min) * (self.upper / 100.0)) + self.b_min

        scalar = ScaleIntensityRange(
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=self.clip,
            dtype=self.dtype,
        )
        img = scalar(img)
        img = convert_to_tensor(img, track_meta=False)
        return img

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        if self.channel_wise:
            img_t = torch.stack([self._normalize(img=d) for d in img_t])  # type: ignore
        else:
            img_t = self._normalize(img=img_t)

        return convert_to_dst_type(img_t, dst=img)[0]


class ScaleIntensityRangePercentilesD(MapTransform):
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

    backend = ScaleIntensityRangePercentiles.backend

    def __init__(
        self,
        keys: KeysCollection,
        lower: float,
        upper: float,
        b_min: float | None,
        b_max: float | None,
        clip: bool = False,
        relative: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRangePercentiles(
            lower, upper, b_min, b_max, clip, relative, channel_wise, dtype
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key + "_quantiles"] = []
            for ch in range(d[key].size(0)):
                d[key][ch] = self.scaler(d[key][ch][None])[0]
                d[key + "_quantiles"].append(
                    torch.Tensor([self.scaler.a_min, self.scaler.a_max])
                )
            d[key + "_quantiles"] = torch.stack(d[key + "_quantiles"], dim=0)
        return d


class ChannelDropoutD(Transform):
    """
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

    def __init__(
        self,
        image_key: str = "image",
        meta_key: str = None,
        min_ch: int = 1,
        max_ch: int = None,
        shuffle_channels: bool = True,
    ) -> None:
        super().__init__()
        self.image_key = image_key
        self.meta_key = meta_key
        self.min = min_ch
        self.max = max_ch
        self.shuffle = shuffle_channels

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        img = d[self.image_key]
        max_ch = self.max or img.size(0)  # assume channel-first
        min_ch = self.min
        orig_ch = img.size(0)
        channels = list(range(orig_ch))
        if self.shuffle:
            shuffle(channels)
        channels = channels[:max_ch]
        # print()
        # print(img.shape)
        img = img[channels, ...]
        keep_ch = randint(min_ch, max_ch)
        # print(img.shape)
        # print(keep_ch, max_ch, orig_ch)
        for i in range(max_ch):
            # print(i)
            if i >= orig_ch:
                # print('pad')
                img = torch.cat([img, torch.zeros_like(img[[0]])], dim=0)
            elif i >= keep_ch:
                img[i] = torch.zeros_like(img[i])
        # print(img.shape)
        if self.meta_key is not None:
            meta = d[self.meta_key]
            # print(meta.shape)
            meta = torch.chunk(meta, orig_ch, dim=0)
            # print(len(meta))
            meta = [meta[ch] for ch in channels]
            # print(len(meta))
            for i in range(max_ch):
                if i >= orig_ch:
                    meta.append(torch.zeros_like(meta[0]))
                elif i >= keep_ch:
                    meta[i] = torch.zeros_like(meta[i])
            meta = torch.cat(meta, dim=0)
            # print(meta.shape)
        d[self.image_key] = img
        d[self.meta_key] = meta
        return d


class ChannelPadD(Transform):
    def __init__(
        self,
        image_key: str = "image",
        meta_key: str = None,
        max_ch: int = 4,
    ) -> None:
        super().__init__()
        self.image_key = image_key
        self.meta_key = meta_key
        self.max = max_ch

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        img = d[self.image_key]
        max_ch = self.max
        orig_ch = img.size(0)
        if orig_ch > max_ch:
            print(
                "WARNING: Too many input channels ({}). Extracting first {} channels and discarding rest.".format(
                    orig_ch, max_ch
                )
            )
            img = img[:max_ch]
            if self.meta_key is not None:
                meta = d[self.meta_key]
                meta = torch.chunk(meta, orig_ch, dim=0)
                meta = meta[:max_ch]
                d[self.meta_key] = meta
        elif orig_ch < max_ch:
            print(
                "Not enough input channels ({}). Padding to total of {} channels..".format(
                    orig_ch, max_ch
                )
            )
            img = torch.cat(
                [img] + int(max_ch - orig_ch) * [torch.zeros_like(img[[0]])], dim=0
            )
            if self.meta_key is not None:
                meta = d[self.meta_key]
                meta = torch.chunk(meta, orig_ch, dim=0)
                meta = torch.cat(
                    list(meta) + int(max_ch - orig_ch) * [torch.zeros_like(meta[0])],
                    dim=0,
                )
                d[self.meta_key] = meta
        d[self.image_key] = img
        return d


from sklearn import mixture


class GMMAugmentD(Transform):
    def __init__(
        self,
        image_key: str = "image",
        mask_key: str = None,
    ) -> None:
        super().__init__()
        self.image_key = image_key
        self.mask_key = mask_key

    def normalize_image(self, image, clip_percentiles=False, pmin=1, pmax=99):
        if clip_percentiles is True:
            pmin = np.percentile(image, pmin)
            pmax = np.percentile(image, pmax)
            v = np.clip(image, pmin, pmax)
        else:
            v = image.copy()

        v_min = v.min(axis=(0, 1, 2), keepdims=True)
        v_max = v.max(axis=(0, 1, 2), keepdims=True)

        return (v - v_min) / (v_max - v_min)

    def select_component_size(self, x, min_components=1, max_components=5):
        lowest_bic = np.infty
        bic = []
        n_components_range = range(min_components, max_components)
        cv_type = "full"  # covariance type for the GaussianMixture function
        best_gmm = None
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(x)
            bic.append(gmm.aic(x))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        return best_gmm.n_components, best_gmm

    def fit_gmm(self, x, n_components=3):
        if n_components is None:
            n_components, gmm = self.select_component_size(
                x, min_components=3, max_components=7
            )
        else:
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type="diag", tol=1e-3
            )

        return gmm.fit(x)

    def get_new_components(
        self, gmm, p_mu=None, q_sigma=None, std_means=None, std_sigma=None
    ):
        sort_indices = gmm.means_[:, 0].argsort(axis=0)
        mu = np.array(gmm.means_[:, 0][sort_indices])
        std = np.array(np.sqrt(gmm.covariances_[:, 0])[sort_indices])

        n_components = mu.shape[0]

        if std_means is not None:
            # use pre-computed intervals to draw values for each component in the mixture
            rng = np.random.default_rng()
            if p_mu is None:
                var_mean_diffs = np.array(std_means)
                p_mu = rng.uniform(-var_mean_diffs, var_mean_diffs)
        if std_sigma is not None:
            rng = np.random.default_rng()
            if q_sigma is None:
                var_std_diffs = np.array(std_sigma)
                q_sigma = rng.uniform(-var_std_diffs, var_std_diffs)
        else:
            # Draw random values for each component in the mixture
            # Multiply by random int for shifting left (-1), right (1) or not changing (0) the parameter.
            if p_mu is None:
                p_mu = (
                    # 0.06
                    0.5
                    * np.random.random(n_components)
                    * np.random.randint(-1, 2, n_components)
                )
            if q_sigma is None:
                q_sigma = (
                    # 0.005
                    0.1
                    * np.random.random(n_components)
                    * np.random.randint(-1, 2, n_components)
                )

        new_mu = mu + p_mu
        new_std = std + q_sigma

        return {"mu": mu, "std": std, "new_mu": new_mu, "new_std": new_std}

    def reconstruct_intensities(self, data, dict_parameters):
        mu, std = dict_parameters["mu"], dict_parameters["std"]
        new_mu, new_std = dict_parameters["new_mu"], dict_parameters["new_std"]
        n_components = len(mu)

        # if we know the values of mean (mu) and standard deviation (sigma) we can find the new value of a voxel v
        # Fist we find the value of a factor w that informs about the percentile a given pixel belongs to: mu*v = d*sigma
        d_im = np.zeros(((n_components,) + data.shape))
        for k in range(n_components):
            d_im[k] = (data.ravel() - mu[k]) / (std[k] + 1e-7)

        # we force the new pixel intensity to lie within the same percentile in the new distribution as in the original
        # distribution: px = mu + d*sigma
        intensities_im = np.zeros(((n_components,) + data.shape))
        for k in range(n_components):
            intensities_im[k] = new_mu[k] + d_im[k] * new_std[k]

        return intensities_im

    def get_new_image_composed(self, intensities_im, probas_original):
        n_components = probas_original.shape[1]
        new_image_composed = np.zeros(intensities_im[0].shape)
        for k in range(n_components):
            new_image_composed = (
                new_image_composed + probas_original[:, k] * intensities_im[k]
            )

        return new_image_composed

    def generate_gmm_image(
        self,
        image,
        mask=None,
        n_components=None,
        q_sigma=None,
        p_mu=None,
        std_means=None,
        std_sigma=None,
        normalize=True,
        percentiles=True,
    ):
        if normalize:
            image = self.normalize_image(
                image, clip_percentiles=percentiles, pmin=1, pmax=99
            )  # the percentiles can be changed

        if mask is None:
            masked_image = image
        else:
            masked_image = image * mask

        # # we only want nonzero values
        data = masked_image[masked_image > 0]
        x = np.expand_dims(data, 1)

        gmm = self.fit_gmm(x, n_components)
        sort_indices = gmm.means_[:, 0].argsort(axis=0)

        # Estimate the posterior probabilities
        probas_original = gmm.predict_proba(x)[:, sort_indices]

        # Get the new intensity components
        params_dict = self.get_new_components(
            gmm, p_mu=p_mu, q_sigma=q_sigma, std_means=std_means, std_sigma=std_sigma
        )
        intensities_im = self.reconstruct_intensities(data, params_dict)

        # Then we add the three predicted images by taking into consideration the probability that each pixel belongs to a
        # certain component of the gaussian mixture (probas_original)
        new_image_composed = self.get_new_image_composed(
            intensities_im, probas_original
        )

        # Reconstruct the image
        new_image = np.zeros(image.shape)
        new_image[np.where(masked_image > 0)] = new_image_composed

        # Put the skull back
        new_image[np.where(masked_image == 0)] = image[np.where(masked_image == 0)]

        # Return the image in [0,1]
        new_image = self.normalize_image(new_image, clip_percentiles=False)

        return new_image

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        img = d[self.image_key]
        mask = d[self.mask_key].cpu().numpy() if self.mask_key is not None else None
        img_cpu = img.cpu().numpy()
        img_norm = self.normalize_image(img_cpu, clip_percentiles=True, pmin=1, pmax=99)
        img_aug = self.generate_gmm_image(img_norm, mask, n_components=8)
        img = torch.from_numpy(img_aug).to(img.device)
        d[self.image_key] = img
        return d