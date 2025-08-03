import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from math import exp
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()

class LAPSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unaries: torch.Tensor, params: dict):
        device = unaries.device
        labelling = torch.zeros_like(unaries)
        unaries_np = unaries.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(unaries_np)
        labelling[row_ind, col_ind] = 1.
        ctx.labels = labelling
        ctx.col_labels = col_ind
        ctx.params = params
        ctx.unaries = unaries  # save unaries for backward
        ctx.device = device
        return labelling.to(device)

    @staticmethod
    def backward(ctx, unary_gradients: torch.Tensor):
        assert ctx.unaries.shape == unary_gradients.shape
        lambda_val = ctx.params.get("lambda", 1.0)
        epsilon_val = 1e-6
        unaries = ctx.unaries
        device = unary_gradients.device

        unaries_np_monitor = unaries.detach().cpu().numpy()
        unary_gradients_np_monitor = unary_gradients.detach().cpu().numpy()
        lambda_times_grad_np_monitor = (lambda_val * unary_gradients).detach().cpu().numpy()

        logger.debug(f"===== LAPSolver Backward Monitoring (Post-Normalization) =====")
        logger.debug(f"Lambda value: {lambda_val:.4f}")
        logger.debug(f"Unaries (cost matrix) stats:")
        logger.debug(f"  Mean: {unaries_np_monitor.mean():.4e}, Std: {unaries_np_monitor.std():.4e}")
        logger.debug(f"Unary gradients (dL/dy) stats:")
        logger.debug(f"  Mean: {unary_gradients_np_monitor.mean():.4e}, Std: {unary_gradients_np_monitor.std():.4e}")
        logger.debug(f"Lambda * Gradients (λ * dL/dy) stats:")
        logger.debug(
            f"  Mean: {lambda_times_grad_np_monitor.mean():.4e}, Std: {lambda_times_grad_np_monitor.std():.4e}")
        avg_abs_unaries = np.mean(np.abs(unaries_np_monitor))
        avg_abs_lambda_grad = np.mean(np.abs(lambda_times_grad_np_monitor))
        logger.debug(
            f"Comparison: Avg Abs(Unaries) = {avg_abs_unaries:.4e}, Avg Abs(Lambda*Grad) = {avg_abs_lambda_grad:.4e}")
        logger.debug(f"Ratio (Unaries / Lambda*Grad): {avg_abs_unaries / (avg_abs_lambda_grad + epsilon_val):.4f}")
        logger.debug(f"===========================================================")

        unaries_prime = unaries + lambda_val * unary_gradients
        unaries_prime_np = unaries_prime.detach().cpu().numpy()

        bwd_labels = torch.zeros_like(unaries)
        row_ind, col_ind = linear_sum_assignment(unaries_prime_np)
        bwd_labels[row_ind, col_ind] = 1.

        forward_labels = ctx.labels

        unary_grad_bwd = -(forward_labels - bwd_labels) / (lambda_val + epsilon_val)

        return unary_grad_bwd.to(ctx.device), None


def compute_distance_matrix(A_flat, B_flat, distance_type="MSE", chunk_size=128):
    if distance_type == "L1":
        num_A, dim = A_flat.shape
        num_B = B_flat.shape[0]
        device = A_flat.device
        dist_matrix = torch.empty((num_A, num_B), device=device)

        for i in range(0, num_A, chunk_size):
            A_chunk = A_flat[i:i + chunk_size]
            for j in range(0, num_B, chunk_size):
                B_chunk = B_flat[j:j + chunk_size]
                A_exp = A_chunk[:, None, :]
                B_exp = B_chunk[None, :, :]
                dist = torch.abs(A_exp - B_exp).sum(dim=2)
                dist_matrix[i:i + A_chunk.size(0), j:j + B_chunk.size(0)] = dist / dim

        return dist_matrix

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

    elif distance_type == "Cosine":
        sim_matrix = F.cosine_similarity(A_flat[:, None, :], B_flat[None, :, :], dim=-1)
        return 1 - sim_matrix

    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")


def _normalize_cost_matrix(cost_matrix_raw):
    """
    normalized to [0, 1]
    """
    min_val = torch.min(cost_matrix_raw)
    max_val = torch.max(cost_matrix_raw)
    if max_val > min_val:
        cost_matrix = (cost_matrix_raw - min_val) / (max_val - min_val)
    else:
        cost_matrix = cost_matrix_raw
        logger.debug("Cost matrix has uniform values, normalization skipped.")
    return cost_matrix


class DifferentiableHungarianLoss(nn.Module):
    def __init__(self, distance_type="MSE", lambda_val=1.0):
        super().__init__()
        self.distance_type = distance_type
        self.lambda_val = lambda_val

    def forward(self, latent, inv_perm_A=None, inv_perm_B=None):
        assert latent.shape[0] == 2, "Latent input must be shape (2, N, ...)"
        num_cells = latent.shape[1]
        latent_dim = latent.shape[2:].numel()

        latent_A = latent[0].view(num_cells, latent_dim)
        latent_B = latent[1].view(num_cells, latent_dim)

        cost_matrix_raw = compute_distance_matrix(latent_A, latent_B, self.distance_type)


        cost_matrix = _normalize_cost_matrix(cost_matrix_raw)

        params = {"lambda": self.lambda_val}
        predicted_matching = LAPSolver.apply(cost_matrix, params)  # use normalized cost matrix

        ideal_matching = torch.zeros_like(predicted_matching)
        ideal_matching[inv_perm_A, inv_perm_B] = 1.0

        loss = HammingLoss()(predicted_matching, ideal_matching)

        col_ind = predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(num_cells)

        return loss, (row_ind, col_ind)


class MultiLayerHungarianLoss(nn.Module):
    def __init__(self, layer_weights, distance_type="MSE", lambda_val=1.0):
        super().__init__()
        self.layer_weights = layer_weights
        self.distance_type = distance_type
        self.lambda_val = lambda_val

    def forward(self, multi_layer_latents, inv_perm_A=None, inv_perm_B=None):
        assert len(multi_layer_latents) == len(self.layer_weights), \
            "Number of latent layers and weights must match"
        assert all(latent.shape[0] == 2 for latent in multi_layer_latents), \
            "Each latent tensor must have shape (2, N, ...)"

        N = multi_layer_latents[0].shape[1]
        device = multi_layer_latents[0].device

        total_loss = 0
        combined_cost_matrix = torch.zeros((N, N), device=device)
        params = {"lambda": self.lambda_val}

        for weight, latent in zip(self.layer_weights, multi_layer_latents):
            latent_A = latent[0].view(N, -1).to(device)
            latent_B = latent[1].view(N, -1).to(device)

            cost_raw = compute_distance_matrix(latent_A, latent_B, self.distance_type)

            cost = _normalize_cost_matrix(cost_raw)

            combined_cost_matrix += weight * cost

            predicted_matching_layer = LAPSolver.apply(cost, params)
            ideal_matching = torch.zeros_like(predicted_matching_layer)
            ideal_matching[inv_perm_A, inv_perm_B] = 1.0
            loss_layer = HammingLoss()(predicted_matching_layer, ideal_matching)
            total_loss += weight * loss_layer
            # ----------------------------------------------------

        final_cost_matrix = _normalize_cost_matrix(combined_cost_matrix)
        # ====================================================

        final_predicted_matching = LAPSolver.apply(final_cost_matrix, params)

        # --- If you only want to calculate the final loss, you can use the following code to replace the above loss calculation in the loop ---
        # ideal_matching = torch.zeros_like(final_predicted_matching)
        # ideal_matching[inv_perm_A, inv_perm_B] = 1.0
        # total_loss = HammingLoss()(final_predicted_matching, ideal_matching)
        # -----------------------------------------------------------------

        col_ind = final_predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(N)

        return total_loss, (row_ind, col_ind)


def build_loss(args):
    lambda_default = 1.0

    if args["USE_MULTI_LAYER_MATCHING"]:
        return MultiLayerHungarianLoss(
            layer_weights=args.get("LAYER_WEIGHTS", [0.5, 0.5]),
            distance_type=args.get("DISTANCE_TYPE", "MSE"),
            lambda_val=args.get("LAMBDA", lambda_default)
        )
    else:
        return DifferentiableHungarianLoss(
            distance_type=args.get("DISTANCE_TYPE", "MSE"),
            lambda_val=args.get("LAMBDA", lambda_default)
        )
