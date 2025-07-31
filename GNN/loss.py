import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
# from math import exp # ### MODIFIED ### 移除了未使用的 import

# ### NEW ###
import logging  # 引入 logging

logging.basicConfig(level=logging.DEBUG,  # 將級別設定為 DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
# ==============================================================================



class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


class LAPSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unaries: torch.Tensor, params: dict):
        device = unaries.device

        # ### MODIFIED ### 為了防止 ctx.unaries 被覆蓋，將其明確保存
        # ctx.save_for_backward(unaries) # 這會將 unaries 作為 saved_tensors[0]
        ctx.unaries_orig = unaries.clone()  # 保存原始 unaries 的副本

        labelling = torch.zeros_like(unaries)
        unaries_np = unaries.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(unaries_np)
        labelling[row_ind, col_ind] = 1.

        ctx.labels = labelling
        ctx.params = params
        ctx.device = device

        return labelling.to(device)

    @staticmethod
    def backward(ctx, unary_gradients: torch.Tensor):
        # ### MODIFIED ### 從 ctx.unaries_orig 獲取原始 unaries
        unaries_original_for_bwd = ctx.unaries_orig

        assert unaries_original_for_bwd.shape == unary_gradients.shape, \
            f"Shape mismatch: unaries {unaries_original_for_bwd.shape} vs gradients {unary_gradients.shape}"

        lambda_val = ctx.params.get("lambda", 1.0)
        epsilon_val = 1e-6  # 避免除以零

        # ======================================================================
        # 日誌監控部分：類似於 paste.txt 的輸出
        unaries_np_monitor = unaries_original_for_bwd.detach().cpu().numpy()
        unary_gradients_np_monitor = unary_gradients.detach().cpu().numpy()
        lambda_times_grad_np_monitor = (lambda_val * unary_gradients).detach().cpu().numpy()

        logger.debug(f"===== LAPSolver Backward Monitoring =====")
        logger.debug(f"Lambda value: {lambda_val:.4f}")
        logger.debug(f"Unaries (cost matrix) stats (before perturbation):")
        logger.debug(f"  Min: {unaries_np_monitor.min():.4e}, Max: {unaries_np_monitor.max():.4e}, "
                     f"Mean: {unaries_np_monitor.mean():.4e}, Std: {unaries_np_monitor.std():.4e}")

        logger.debug(f"Unary gradients (dL/dy) stats:")
        logger.debug(f"  Min: {unary_gradients_np_monitor.min():.4e}, Max: {unary_gradients_np_monitor.max():.4e}, "
                     f"Mean: {unary_gradients_np_monitor.mean():.4e}, Std: {unary_gradients_np_monitor.std():.4e}")

        logger.debug(f"Lambda * Gradients (lambda * dL/dy) stats:")
        logger.debug(f"  Min: {lambda_times_grad_np_monitor.min():.4e}, Max: {lambda_times_grad_np_monitor.max():.4e}, "
                     f"Mean: {lambda_times_grad_np_monitor.mean():.4e}, Std: {lambda_times_grad_np_monitor.std():.4e}")

        avg_abs_unaries = np.mean(np.abs(unaries_np_monitor))
        avg_abs_lambda_grad = np.mean(np.abs(lambda_times_grad_np_monitor))

        logger.debug(
            f"Comparison: Avg Abs(Unaries) = {avg_abs_unaries:.4e}, Avg Abs(Lambda*Grad) = {avg_abs_lambda_grad:.4e}")

        # 避免除以零
        ratio = avg_abs_unaries / (avg_abs_lambda_grad + epsilon_val)
        logger.debug(f"Ratio (Avg Abs(Unaries) / Avg Abs(Lambda*Grad)): {ratio:.4f}")
        logger.debug(f"=========================================")
        # ======================================================================

        # w′ = w + λ ∇L/∇y
        # 關鍵修正：確保擾動項的影響力
        unaries_prime = unaries_original_for_bwd + lambda_val * unary_gradients
        unaries_prime_np = unaries_prime.detach().cpu().numpy()

        bwd_labels = torch.zeros_like(unaries_original_for_bwd)
        row_ind, col_ind = linear_sum_assignment(unaries_prime_np)
        bwd_labels[row_ind, col_ind] = 1.

        forward_labels = ctx.labels

        # ∇fλ(w) = −(ŷ − yλ) / λ
        unary_grad_bwd = -(forward_labels - bwd_labels) / (lambda_val + epsilon_val)

        return unary_grad_bwd.to(ctx.device), None


def compute_distance_matrix(A_flat, B_flat, distance_type="MSE"):  # ### MODIFIED ### 移除了未使用的 chunk_size
    if distance_type == "L1":
        # ... (L1 logic remains unchanged) ...
        # 這部分程式碼在您提供的前幾版中未完整，請根據需要填寫
        num_A, dim = A_flat.shape
        num_B = B_flat.shape[0]
        device = A_flat.device
        dist_matrix = torch.empty((num_A, num_B), device=device)
        for i in range(num_A):
            dist_matrix[i, :] = torch.sum(torch.abs(A_flat[i, :] - B_flat), dim=1)
        return dist_matrix / dim
    elif distance_type == "MSE":
        A_sq = torch.sum(A_flat ** 2, dim=1, keepdim=True)
        B_sq = torch.sum(B_flat ** 2, dim=1, keepdim=True)
        AB = torch.matmul(A_flat, B_flat.transpose(0, 1))

        A_sq = A_sq.expand_as(AB)
        B_sq = B_sq.transpose(0, 1).expand_as(AB)

        distance_sq = A_sq - 2 * AB + B_sq
        distance_sq = torch.clamp(distance_sq, min=0)

        latent_dim = A_flat.shape[1]
        mse_matrix = distance_sq / latent_dim
        return mse_matrix
    # ### NEW ###
    elif distance_type == "Cosine":
        sim_matrix = F.cosine_similarity(A_flat[:, None, :], B_flat[None, :, :], dim=-1)
        return 1 - sim_matrix
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")


class DifferentiableHungarianLoss(nn.Module):
    # ### MODIFIED ### 將默認值從 20 改為 1.0
    def __init__(self, distance_type="MSE", lambda_val=1.0):
        super().__init__()
        self.distance_type = distance_type
        self.lambda_val = lambda_val

    def forward(self, latent, inv_perm_A=None, inv_perm_B=None):
        assert latent.shape[0] == 2, "Latent input must be shape (2, N, ...)"

        num_cells = latent.shape[1]
        latent_dim = latent.shape[2:].numel()

        latent_A = latent[0]
        latent_B = latent[1]

        latent_A = latent_A.view(num_cells, latent_dim)
        latent_B = latent_B.view(num_cells, latent_dim)

        cost_matrix_raw = compute_distance_matrix(latent_A, latent_B, self.distance_type)

        # === MODIFIED: 關鍵修正：對成本矩陣進行 Min-Max 標準化 ===
        # 確保成本矩陣的數值範圍穩定在 [0, 1] 之間，
        # 使得 lambda_val 真正能控制擾動的相對大小。
        min_val = cost_matrix_raw.min()
        max_val = cost_matrix_raw.max()
        if max_val > min_val:
            cost_matrix = (cost_matrix_raw - min_val) / (max_val - min_val)
        else:
            # 如果所有值都一樣 (例如，所有節點距離都為0)，則不變，避免除以零
            cost_matrix = cost_matrix_raw
            logging.warning("所有節點距離都相同，成本矩陣未標準化。")
        # ================================================

        params = {"lambda": self.lambda_val}
        predicted_matching = LAPSolver.apply(cost_matrix, params)  # 使用標準化後的成本矩陣

        ideal_matching = torch.zeros_like(predicted_matching)
        # ### MODIFIED ### 確保 inv_perm_A, inv_perm_B 是 Long Tensor
        ideal_matching[inv_perm_A.long(), inv_perm_B.long()] = 1.0

        loss = HammingLoss()(predicted_matching, ideal_matching)

        col_ind = predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(num_cells)

        return loss, (row_ind, col_ind)