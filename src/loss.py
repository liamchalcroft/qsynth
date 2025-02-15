import torch
from torch import cat
from torch import tensor, Tensor
from torch import reshape
from torch.nn.modules.loss import _Loss
import numpy as np


### Adapted from https://github.com/danieltudosiu/nmpevqvae/blob/master/losses.py


def log1p_safe(x):
    """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
    x = torch.as_tensor(x)
    return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))


def expm1_safe(x):
    """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
    x = torch.as_tensor(x)
    return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))


def barron_lossfun(x, alpha, scale, approximate=False, epsilon=1e-6):
    r"""Implements the general form of the loss.

    This implements the rho(x, \alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.

    Args:
      x: The residual for which the loss is being computed. x can have any shape,
        and alpha and scale will be broadcasted to match x's shape if necessary.
        Must be a tensor of floats.
      alpha: The shape parameter of the loss (\alpha in the paper), where more
        negative values produce a loss with more robust behavior (outliers "cost"
        less), and more positive values produce a loss with less robust behavior
        (outliers are penalized more heavily). Alpha can be any value in
        [-infinity, infinity], but the gradient of the loss with respect to alpha
        is 0 at -infinity, infinity, 0, and 2. Must be a tensor of floats with the
        same precision as `x`. Varying alpha allows
        for smooth interpolation between a number of discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
      scale: The scale parameter of the loss. When |x| < scale, the loss is an
        L2-like quadratic bowl, and when |x| > scale the loss function takes on a
        different shape according to alpha. Must be a tensor of single-precision
        floats.
      approximate: a bool, where if True, this function returns an approximate and
        faster form of the loss, as described in the appendix of the paper. This
        approximation holds well everywhere except as x and alpha approach zero.
      epsilon: A float that determines how inaccurate the "approximate" version of
        the loss will be. Larger values are less accurate but more numerically
        stable. Must be great than single-precision machine epsilon.

    Returns:
      The losses for each element of x, in the same shape and precision as x.
    """
    assert torch.is_tensor(x)
    assert torch.is_tensor(scale)
    assert torch.is_tensor(alpha)
    assert alpha.dtype == x.dtype
    assert scale.dtype == x.dtype
    assert (scale > 0).all()
    if approximate:
        # `epsilon` must be greater than single-precision machine epsilon.
        assert epsilon > np.finfo(np.float32).eps
        # Compute an approximate form of the loss which is faster, but innacurate
        # when x and alpha are near zero.
        b = torch.abs(alpha - 2) + epsilon
        d = torch.where(alpha >= 0, alpha + epsilon, alpha - epsilon)
        loss = (b / d) * (torch.pow((x / scale) ** 2 / b + 1.0, 0.5 * d) - 1.0)
    else:
        # Compute the exact loss.

        # This will be used repeatedly.
        squared_scaled_x = (x / scale) ** 2

        # The loss when alpha == 2.
        loss_two = 0.5 * squared_scaled_x
        # The loss when alpha == 0.
        loss_zero = log1p_safe(0.5 * squared_scaled_x)
        # The loss when alpha == -infinity.
        loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
        # The loss when alpha == +infinity.
        loss_posinf = expm1_safe(0.5 * squared_scaled_x)

        # The loss when not in one of the above special cases.
        machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(x)
        # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
        beta_safe = torch.max(machine_epsilon, torch.abs(alpha - 2.0))
        # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
        alpha_safe = torch.where(
            alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha)
        ) * torch.max(machine_epsilon, torch.abs(alpha))
        loss_otherwise = (beta_safe / alpha_safe) * (
            torch.pow(squared_scaled_x / beta_safe + 1.0, 0.5 * alpha) - 1.0
        )

        # Select which of the cases of the loss to return.
        loss = torch.where(
            alpha == -float("inf"),
            loss_neginf,
            torch.where(
                alpha == 0,
                loss_zero,
                torch.where(
                    alpha == 2,
                    loss_two,
                    torch.where(alpha == float("inf"), loss_posinf, loss_otherwise),
                ),
            ),
        )

    return loss


class BarronLoss(_Loss):
    def __init__(
        self,
        alpha=1,
        scale=1,
        approximate=False,
        epsilon=1e-6,
        size_average=None,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.scale = scale
        self.approximate = approximate
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = barron_lossfun(input - target)
        return (
            loss.mean()
            if self.reduction == "mean"
            else loss.sum() if self.reduction == "sum" else loss
        )


class BaurLoss(_Loss):
    def __init__(
        self,
        recon_weight=1.0,
        grad_weight=1.0,
        size_average=None,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(size_average, reduce, reduction)
        self.l1_loss = torch.nn.functional.l1_loss
        self.l2_loss = torch.nn.functional.mse_loss
        self.recon_weight = recon_weight
        self.grad_weight = grad_weight
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        l1_recon = (
            self.l1_loss(input, target, reduction=self.reduction) * self.recon_weight
        )
        l2_recon = (
            self.l2_loss(input, target, reduction=self.reduction) * self.recon_weight
        )

        target_gradients = self.__image_gradients(target)
        input_gradients = self.__image_gradients(input)

        l1_grad = (
            self.l1_loss(
                target_gradients[0], input_gradients[0], reduction=self.reduction
            )
            + self.l1_loss(
                target_gradients[1], input_gradients[1], reduction=self.reduction
            )
            + self.l1_loss(
                target_gradients[2], input_gradients[2], reduction=self.reduction
            )
        ) * self.grad_weight

        l2_grad = (
            self.l2_loss(
                target_gradients[0], input_gradients[0], reduction=self.reduction
            )
            + self.l2_loss(
                target_gradients[1], input_gradients[1], reduction=self.reduction
            )
            + self.l2_loss(
                target_gradients[2], input_gradients[2], reduction=self.reduction
            )
        ) * self.grad_weight

        loss_total = l1_recon + l2_recon + l1_grad + l2_grad

        return loss_total

    @staticmethod
    def __image_gradients(image):
        input_shape = image.shape
        batch_size, features, depth, height, width = input_shape

        dz = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dx = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]

        dzz = tensor(()).new_zeros(
            (batch_size, features, 1, height, width),
            device=image.device,
            dtype=dz.dtype,
        )
        dz = cat([dz, dzz], 2)
        dz = reshape(dz, input_shape)

        dyz = tensor(()).new_zeros(
            (batch_size, features, depth, 1, width), device=image.device, dtype=dy.dtype
        )
        dy = cat([dy, dyz], 3)
        dy = reshape(dy, input_shape)

        dxz = tensor(()).new_zeros(
            (batch_size, features, depth, height, 1),
            device=image.device,
            dtype=dx.dtype,
        )
        dx = cat([dx, dxz], 4)
        dx = reshape(dx, input_shape)

        return dx, dy, dz
  