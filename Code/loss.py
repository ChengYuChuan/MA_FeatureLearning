import torch
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils import get_logger
import math
import warnings


logger = get_logger('Loss')

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes Dice Coefficient as defined in https://arxiv.org/abs/1606.04797 for multi-channel input and target.
    Assumes the input is a normalized probability, e.g., output of Sigmoid or Softmax.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target).float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

# ------------------SSIM----------------------

def create_gaussian_kernel_3d(kernel_size=7, sigma=1.5, channels=1):
    """
    Create a 3D Gaussian kernel
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    gaussian_kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)  # Supports multiple channels
    return gaussian_kernel

class SSIM3D(nn.Module):
    def __init__(self, window_size=7, sigma=1.5, C1=0.01**2, C2=0.03**2, use_gaussian=True):
        """
        3D SSIM loss function. Supports Gaussian kernel or average pooling.

        Args:
            window_size: local window size
            sigma: standard deviation for Gaussian smoothing (only used if use_gaussian=True)
            C1, C2: stability terms for SSIM computation
            use_gaussian: whether to use Gaussian kernel (default True)
        """
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        self.sigma = sigma
        self.use_gaussian = use_gaussian
        self.padding = window_size // 2

        # 预计算 Gaussian Kernel
        if use_gaussian:
            self.gaussian_kernel = create_gaussian_kernel_3d(window_size, sigma)

    def compute_mu_sigma(self, x):
        """
        Compute local mean (mu) and variance (sigma)
        """
        if self.use_gaussian:
            gaussian_kernel = self.gaussian_kernel.to(device=x.device, dtype=x.dtype)
            mu = F.conv3d(x, gaussian_kernel, stride=1, padding=self.padding, groups=x.shape[1])
            sigma = F.conv3d(x ** 2, gaussian_kernel, stride=1, padding=self.padding, groups=x.shape[1]) - mu ** 2
            sigma = torch.clamp(sigma, min=0.0) # make sure that it won't be negative
        else:
            mu = F.avg_pool3d(x, kernel_size=self.window_size, stride=1, padding=self.padding)
            sigma = F.avg_pool3d(x ** 2, kernel_size=self.window_size, stride=1, padding=self.padding) - mu ** 2
            sigma = torch.clamp(sigma, min=0.0)  # make sure that it won't be negative
        return mu, sigma

    def forward(self, x, y):
        """
        Compute 3D SSIM
        """
        x = x.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        mu_x, sigma_x = self.compute_mu_sigma(x)
        mu_y, sigma_y = self.compute_mu_sigma(y)
        sigma_xy = self.compute_mu_sigma(x * y)[0] - mu_x * mu_y

        # compute SSIM
        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        den = torch.clamp(den, min=1e-8)

        SSIM = num / den
        return 1 - SSIM.mean()

# ------------------SSIM END----------------------

class MS_SSIM3D(nn.Module):
    def __init__(self, window_size=5, sigma=1.5, C1=0.01**2, C2=0.03**2,
                 use_gaussian=False, levels=3, weights=None,
                 pool_type='avg', verbose=True):
        super(MS_SSIM3D, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2
        self.use_gaussian = use_gaussian
        self.levels = levels
        self.weights = weights if weights is not None else [1.0 / levels] * levels
        self.pool_type = pool_type
        self.verbose = verbose
        self.padding = window_size // 2

        assert len(self.weights) == self.levels, "Weights length must match levels"
        assert self.pool_type in ['avg', 'max'], "pool_type must be 'avg' or 'max'"

    def _create_kernel(self, x):
        return create_gaussian_kernel_3d(self.window_size, self.sigma, x.shape[1]).to(x.device)

    def _compute_mu_sigma(self, x, kernel):
        if self.use_gaussian:
            mu = F.conv3d(x, kernel, stride=1, padding=self.padding, groups=x.shape[1])
            sigma = F.conv3d(x ** 2, kernel, stride=1, padding=self.padding, groups=x.shape[1]) - mu ** 2
        else:
            mu = F.avg_pool3d(x, kernel_size=self.window_size, stride=1, padding=self.padding)
            sigma = F.avg_pool3d(x ** 2, kernel_size=self.window_size, stride=1, padding=self.padding) - mu ** 2

        sigma = torch.clamp(sigma, min=0.0)
        return mu, sigma

    def _ssim(self, x, y, kernel):
        x = x.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        mu_x, sigma_x = self._compute_mu_sigma(x, kernel)
        mu_y, sigma_y = self._compute_mu_sigma(y, kernel)
        sigma_xy = self._compute_mu_sigma(x * y)[0] - mu_x * mu_y
        sigma_xy = torch.clamp(sigma_xy, min=0.0)

        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        den = torch.clamp(den, min=1e-8)
        return (num / den).mean()

    def forward(self, x, y):
        ms_ssim = []

        for i in range(self.levels):
            kernel = self._create_kernel(x)

            # === Debug: print current layer state ===
            if self.verbose:
                nz_ratio = (x != 0).float().mean().item()
                print(f"Level {i}: min={x.min().item():.5f}, max={x.max().item():.5f}, mean={x.mean().item():.5f}, non-zero={nz_ratio:.5f}")

            # === Sparse detection: If the non-zero ratio is too low, skip the layer ===
            if (x != 0).float().mean().item() < 0.001:
                if self.verbose:
                    print(f"[Skip] Level {i} input too sparse, skipping")
                ssim_i = torch.ones(1, device=x.device)
            else:
                ssim_i = self._ssim(x, y, kernel)

            ms_ssim.append(ssim_i)

            # === downsampling ===
            if i < self.levels - 1:
                if self.pool_type == 'avg':
                    x = F.avg_pool3d(x, kernel_size=2, stride=2, padding=0)
                    y = F.avg_pool3d(y, kernel_size=2, stride=2, padding=0)
                else:  # max pooling
                    x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)
                    y = F.max_pool3d(y, kernel_size=2, stride=2, padding=0)

        ms_ssim = torch.stack(ms_ssim)
        weights = torch.tensor(self.weights, device=ms_ssim.device)
        ms_ssim_weighted = torch.prod(ms_ssim ** weights)

        return 1 - ms_ssim_weighted

class MSEDiceLoss(nn.Module):
    """Linear combination of MSE and Dice losses"""

    def __init__(self, alpha=0.5):
        super(MSEDiceLoss, self).__init__()
        self.alpha = alpha
        self.MSE = nn.MSELoss()
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.MSE(input, target) + (1 - self.alpha) * self.dice(input, target)

class MSESSIMLoss(nn.Module):
    """Linear combination of MSE and SSIM losses"""

    def __init__(self, alpha=0.2, window_size=5):
        super(MSESSIMLoss, self).__init__()
        self.alpha = alpha
        self.MSE = nn.MSELoss()
        self.SSIM = SSIM3D(window_size=window_size)

    def forward(self, input, target):
        return self.alpha * self.MSE(input, target) + (1 - self.alpha) * self.SSIM(input, target)

class L1SSIMLoss(nn.Module):
    """Linear combination of L1 and SSIM 3D losses"""

    def __init__(self, alpha=0.8, window_size=5):
        super(L1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.L1 = nn.L1Loss(reduction='none')  # Warning：use 'none' then you can put mask
        self.SSIM = SSIM3D(window_size=window_size)
        self.last_l1 = 0
        self.last_ssim = 0

    def forward(self, input, target):
        mask = (target > 0).float()
        l1_loss_all = torch.abs(input - target)
        masked_l1_loss = (l1_loss_all * mask).sum() / (mask.sum() + 1e-8)

        ssim_loss = self.SSIM(input, target)
        self.last_l1 = masked_l1_loss.item()
        self.last_ssim = ssim_loss.item()
        return self.alpha * masked_l1_loss + (1 - self.alpha) * ssim_loss

    def debug_log(self, step):
        print(f"[Step {step}] L1 Loss (masked): {self.last_l1:.6f} | SSIM Loss: {self.last_ssim:.6f}")

class HybridL1MSELoss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss(reduction=reduction)
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + (1 - self.alpha) * self.mse(pred, target)

def get_loss_criterion(name, weight=None, ignore_index=None, skip_last_target=False, pos_weight=None, window_size=5, alpha=0.2, **loss_kwargs):
    """
    Returns the loss function based on provided parameters.

    Args:
        name (str): Name of the loss function.
        weight (list or tensor, optional): Class weights.
        ignore_index (int, optional): Index to ignore during loss computation.
        skip_last_target (bool): If True, skip the last target channel.
        pos_weight (tensor, optional): Positive class weight for BCE-type losses.
        window_size (int): Window size for SSIM.
        alpha (float): Linear combination factor for hybrid losses.
        loss_kwargs (dict): Other loss parameters.

    Returns:
        nn.Module: Loss instance.
    """
    logger.info(f"Creating loss function: {name}")

    if weight is not None:
        weight = torch.tensor(weight).float()
        logger.info(f"Using class weights: {weight}")

    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    loss = _create_loss(
        name, weight, ignore_index, pos_weight,
        alpha=alpha, window_size=window_size,
        **loss_kwargs
    )

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss

def _create_loss(name, weight, ignore_index, pos_weight, alpha=0.5, window_size=5, return_msssim=False, **loss_kwargs):
    if name == 'L1Loss':
        reduction = loss_kwargs.get("reduction", "mean")
        logger.info(f"L1 reduction: {reduction}")
        return nn.L1Loss(reduction=reduction)
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'HybridL1MSELoss':
        logger.info(f"Loss Alpha: {alpha}")
        return HybridL1MSELoss()
    elif name == 'MSESSIMLoss':
        logger.info(f"SSIM window Size: {window_size}, Alpha value (on MSE): {alpha}")
        return MSESSIMLoss(alpha=alpha, window_size=window_size)
    elif name == 'L1SSIMLoss':
        logger.info(f"L1 + SSIM Loss | Window Size: {window_size}, Alpha: {alpha}")
        return L1SSIMLoss(alpha=alpha, window_size=window_size)
    elif name == 'SSIMLoss':
        use_gaussian = loss_kwargs.get('use_gaussian', False)
        logger.info(f"SSIM window Size: {window_size}, use_gaussian: {use_gaussian}")
        return SSIM3D(window_size=window_size)
    elif name == 'MS_SSIMLoss':
        levels = loss_kwargs.get("levels", 3)
        weights = loss_kwargs.get("weights", [0.3, 0.3, 0.4])
        pool_type = loss_kwargs.get("pool_type", "avg")
        logger.info(f"MS-SSIM | window_size={window_size}, levels={levels}, weights={weights}, pool_type={pool_type}")
        return MS_SSIM3D(window_size=window_size, levels=levels, weights=weights, pool_type=pool_type)
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
