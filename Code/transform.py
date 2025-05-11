#---Transform---
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve
import torch
import numpy as np

class Standardize:
    """
    Apply Z-score normalization followed by Min-Max normalization to scale the values into [0,1].
    # **para_min_max
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, min_max=True, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise
        self.min_max = min_max  # 是否應用 Min-Max Normalization

    def __call__(self, m):
        # Convert PyTorch tensor to NumPy array
        if isinstance(m, torch.Tensor):
            m = m.numpy()

        # 計算 mean 和 std
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                axes = list(range(m.ndim))
                axes = tuple(axes[1:])  # 平均 across channels
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        # 先做 Z-score normalization
        standardized = (m - mean) / np.clip(std, a_min=self.eps, a_max=None)

        # **新增 Min-Max Normalization**
        if self.min_max:
            min_val = standardized.min()
            max_val = standardized.max()
            standardized = (standardized - min_val) / (max_val - min_val + self.eps)  # 確保 [0,1]

        return standardized

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m

class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
        normalize (bool): zero-one normalization of the input data
    """

    def __init__(self, expand_dims, dtype=np.float32, normalize=False, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype
        self.normalize = normalize

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        # if self.expand_dims and m.ndim == 3:
        #     m = np.expand_dims(m, axis=0)
        if self.expand_dims and m.ndim == 3:
            m = m[np.newaxis, ...]

        if self.normalize:
            # avoid division by zero
            m = (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-10)

        return torch.from_numpy(m.astype(dtype=self.dtype))