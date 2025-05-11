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
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

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
    生成 3D 高斯核 (Gaussian Kernel)
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    gaussian_kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)  # 适用于多个通道
    return gaussian_kernel

class SSIM3D(nn.Module):
    def __init__(self, window_size=7, sigma=1.5, C1=0.01**2, C2=0.03**2, use_gaussian=True):
        """
        3D SSIM 损失函数，支持使用 Gaussian Kernel 或均值池化
        :param window_size: 局部窗口大小
        :param sigma: 高斯平滑标准差（仅在 use_gaussian=True 时生效）
        :param C1, C2: SSIM 计算中的稳定项
        :param use_gaussian: 是否使用 Gaussian Kernel (默认 True)
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
        计算均值 (mu) 和方差 (sigma)
        """
        if self.use_gaussian:
            # 使用高斯卷积计算局部均值
            gaussian_kernel = self.gaussian_kernel.to(device=x.device, dtype=x.dtype)
            mu = F.conv3d(x, gaussian_kernel, stride=1, padding=self.padding, groups=x.shape[1])
            sigma = F.conv3d(x ** 2, gaussian_kernel, stride=1, padding=self.padding, groups=x.shape[1]) - mu ** 2
            sigma = torch.clamp(sigma, min=0.0) # make sure that it won't be negative
        else:
            # 使用均值池化计算局部均值
            mu = F.avg_pool3d(x, kernel_size=self.window_size, stride=1, padding=self.padding)
            sigma = F.avg_pool3d(x ** 2, kernel_size=self.window_size, stride=1, padding=self.padding) - mu ** 2
            sigma = torch.clamp(sigma, min=0.0)  # make sure that it won't be negative
        return mu, sigma

    def forward(self, x, y):
        """
        计算 SSIM-3D
        """
        x = x.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        # 计算局部均值和方差
        mu_x, sigma_x = self.compute_mu_sigma(x)
        mu_y, sigma_y = self.compute_mu_sigma(y)
        sigma_xy = self.compute_mu_sigma(x * y)[0] - mu_x * mu_y

        # 计算 SSIM
        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        den = torch.clamp(den, min=1e-8)

        SSIM = num / den
        # # Step 1: 建立合法 mask（非 NaN）
        # valid_mask = ~torch.isnan(ssim_map)
        #
        # # Step 2: 只對有效值做加權平均
        # masked_ssim = (ssim_map[valid_mask]).mean() if valid_mask.any() else torch.tensor(1.0, device=x.device)

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

        assert len(self.weights) == self.levels, "weights 長度必須與 levels 相符"
        assert self.pool_type in ['avg', 'max'], "pool_type 必須是 'avg' 或 'max'"

    def _create_kernel(self, x):
        return create_gaussian_kernel_3d(self.window_size, self.sigma, x.shape[1]).to(x.device)

    def _compute_mu_sigma(self, x, kernel):
        if self.use_gaussian:
            mu = F.conv3d(x, kernel, stride=1, padding=self.padding, groups=x.shape[1])
            sigma = F.conv3d(x ** 2, kernel, stride=1, padding=self.padding, groups=x.shape[1]) - mu ** 2
        else:
            mu = F.avg_pool3d(x, kernel_size=self.window_size, stride=1, padding=self.padding)
            sigma = F.avg_pool3d(x ** 2, kernel_size=self.window_size, stride=1, padding=self.padding) - mu ** 2

        sigma = torch.clamp(sigma, min=0.0)  # make sure that it won't be negative
        return mu, sigma

    def _ssim(self, x, y, kernel):
        x = x.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        mu_x, sigma_x = self._compute_mu_sigma(x, kernel)
        mu_y, sigma_y = self._compute_mu_sigma(y, kernel)
        # mu_xy = F.conv3d(x * y, kernel, stride=1, padding=self.padding, groups=x.shape[1])
        # sigma_xy = mu_xy - mu_x * mu_y

        sigma_xy = self.compute_mu_sigma(x * y)[0] - mu_x * mu_y
        sigma_xy = torch.clamp(sigma_xy, min=0.0)  # make sure that it won't be negative

        num = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        den = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        den = torch.clamp(den, min=1e-8)
        ssim_map = num / den
        # ssim_map = torch.nan_to_num(ssim_map, nan=0.0, posinf=0.0, neginf=0.0)

        return ssim_map.mean()

    def forward(self, x, y):
        ms_ssim = []

        for i in range(self.levels):
            kernel = self._create_kernel(x)

            # === Debug: 印出當前層資訊 ===
            if self.verbose:
                nz_ratio = (x != 0).float().mean().item()
                print(f"Level {i}: min={x.min().item():.5f}, max={x.max().item():.5f}, mean={x.mean().item():.5f}, non-zero={nz_ratio:.5f}")

            # === 稀疏偵測：若非零比率過低，跳過該層 ===
            if (x != 0).float().mean().item() < 0.001:
                if self.verbose:
                    print(f"[Skip] Level {i} 輸入過於稀疏，略過此層")
                ssim_i = torch.ones(1, device=x.device)
            else:
                ssim_i = self._ssim(x, y, kernel)

            ms_ssim.append(ssim_i)

            # === 下採樣下一層 ===
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

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)

class MSEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=0.5):
        super(MSEDiceLoss, self).__init__()
        self.alpha = alpha
        self.MSE = nn.MSELoss()
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.MSE(input, target) + (1-self.alpha) * self.dice(input, target)

class MSESSIMLoss(nn.Module):
    """Linear combination of MSE and SSIM losses"""

    def __init__(self, alpha=0.2, window_size=5):
        super(MSESSIMLoss, self).__init__()
        self.alpha = alpha
        self.MSE = nn.MSELoss()
        self.SSIM = SSIM3D(window_size=window_size)  # 使用新的 SSIM 類

    def forward(self, input, target):
        return self.alpha * self.MSE(input, target) + (1 - self.alpha) * self.SSIM(input, target)

class L1SSIMLoss(nn.Module):
    """Linear combination of L1 and SSIM 3D losses"""

    def __init__(self, alpha=0.8, window_size=5):
        super(L1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.L1 = nn.L1Loss(reduction='none')  # 注意：使用 'none' 才能加 mask
        self.SSIM = SSIM3D(window_size=window_size)
        self.last_l1 = 0
        self.last_ssim = 0

    def forward(self, input, target):
        # 建立 mask，僅對非零 voxel 進行 L1 loss 計算
        mask = (target > 0).float()

        l1_loss_all = torch.abs(input - target)
        masked_l1_loss = (l1_loss_all * mask).sum() / (mask.sum() + 1e-8)  # 避免除以 0

        ssim_loss = self.SSIM(input, target)

        # 儲存 log 用的 loss 值
        self.last_l1 = masked_l1_loss.item()
        self.last_ssim = ssim_loss.item()

        # return self.alpha * self.L1(input, target) + (1 - self.alpha) * self.SSIM(input, target)
        # 回傳混合 loss（你可自由調整 10 倍的係數）
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
    :param name: (str) Name of the loss function.
    :param weight: (list or tensor, optional) Class weights for the loss.
    :param ignore_index: (int, optional) Index to ignore in loss calculation.
    :param skip_last_target: (bool) Whether to skip the last target channel.
    :param pos_weight: (tensor, optional) Positive class weight (for BCE-based losses).
    :param loss_kwargs: (dict) Additional keyword arguments for loss functions.
    :return: An instance of the loss function.
    """

    logger.info(f"Creating loss function: {name}")

    if weight is not None:
        weight = torch.tensor(weight).float()
        logger.info(f"Using class weights: {weight}")

    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    # 移除 loss_kwargs 內的 return_msssim，避免重複
    # return_msssim = loss_kwargs.pop("return_msssim", False)

    loss = _create_loss(
        name, weight, ignore_index, pos_weight,
        alpha=alpha, window_size=window_size,
        **loss_kwargs
    )

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss

def _create_loss(name, weight, ignore_index, pos_weight, alpha=0.5, window_size=5, return_msssim=False, **loss_kwargs):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'L1Loss':
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
        return MS_SSIM3D(window_size=window_size, levels=loss_kwargs.get("levels", 2), weights=loss_kwargs.get("weights"), pool_type=loss_kwargs.get("pool_type", "avg"))
    elif name == 'MSEDiceLoss':
        return MSEDiceLoss(alpha=alpha)
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")