import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from math import exp

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
        ctx.unaries = unaries
        ctx.device = device
        return labelling.to(device)

    @staticmethod
    def backward(ctx, unary_gradients: torch.Tensor):
        assert ctx.unaries.shape == unary_gradients.shape

        lambda_val = ctx.params.get("lambda", 15)
        epsilon_val = 1e-6

        unaries = ctx.unaries
        device = unary_gradients.device

        # w′ = w + λ ∇L/∇y
        unaries_prime = unaries + lambda_val * unary_gradients
        unaries_prime_np = unaries_prime.detach().cpu().numpy()

        # yλ = Solver(w′)
        bwd_labels = torch.zeros_like(unaries)
        row_ind, col_ind = linear_sum_assignment(unaries_prime_np)
        bwd_labels[row_ind, col_ind] = 1.

        forward_labels = ctx.labels

        # ∇fλ(w) = −(ŷ − yλ) / λ
        unary_grad_bwd = -(forward_labels - bwd_labels) / (lambda_val + epsilon_val)

        return unary_grad_bwd.to(ctx.device), None

def compute_distance_matrix(A_flat, B_flat, distance_type="MSE"):
    if distance_type == "L1":
        num_A, dim = A_flat.shape
        num_B = B_flat.shape[0]
        device = A_flat.device
        dist_matrix = torch.empty((num_A, num_B), device=device)

        for i in range(0, num_A, chunk_size):
            A_chunk = A_flat[i:i + chunk_size]  # (chunk, dim)
            for j in range(0, num_B, chunk_size):
                B_chunk = B_flat[j:j + chunk_size]  # (chunk, dim)
                # Broadcasting-safe computation
                A_exp = A_chunk[:, None, :]  # (chunkA, 1, dim)
                B_exp = B_chunk[None, :, :]  # (1, chunkB, dim)
                dist = torch.abs(A_exp - B_exp).sum(dim=2)  # (chunkA, chunkB)
                dist_matrix[i:i + A_chunk.size(0), j:j + B_chunk.size(0)] = dist / dim

        return dist_matrix

    elif distance_type == "MSE":
        # Optimized MSE calculation to avoid large intermediate tensor
        # ||a - b||^2 = ||a||^2 - 2a·b + ||b||^2
        # MSE = ||a - b||^2 / latent_dim
        A_sq = torch.sum(A_flat**2, dim=1, keepdim=True) # Shape: (num_cells, 1)
        B_sq = torch.sum(B_flat**2, dim=1, keepdim=True) # Shape: (num_cells, 1)
        AB = torch.matmul(A_flat, B_flat.transpose(0, 1)) # Shape: (num_cells, num_cells)

        # Expand A_sq and B_sq for broadcasting
        A_sq = A_sq.expand_as(AB) # Shape: (num_cells, num_cells)
        B_sq = B_sq.transpose(0, 1).expand_as(AB) # Shape: (num_cells, num_cells)

        distance_sq = A_sq - 2 * AB + B_sq # Shape: (num_cells, num_cells)
        # Ensure non-negativity due to floating point inaccuracies
        distance_sq = torch.clamp(distance_sq, min=0)

        latent_dim = A_flat.shape[1]
        mse_matrix = distance_sq / latent_dim # Shape: (num_cells, num_cells)
        return mse_matrix

    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")


class DifferentiableHungarianLoss(nn.Module):
    def __init__(self, distance_type="MSE", lambda_val=20):
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

        cost_matrix = compute_distance_matrix(latent_A, latent_B, self.distance_type)

        params = {"lambda": self.lambda_val}
        predicted_matching = LAPSolver.apply(cost_matrix, params)

        ideal_matching = torch.zeros_like(predicted_matching)
        ideal_matching[inv_perm_A, inv_perm_B] = 1.0

        loss = HammingLoss()(predicted_matching, ideal_matching)

        col_ind = predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(num_cells)

        return loss, (row_ind, col_ind)

class MultiLayerHungarianLoss(nn.Module):
    def __init__(self, layer_weights, distance_type="MSE", lambda_val=20):
        super().__init__()
        self.layer_weights = layer_weights
        self.distance_type = distance_type
        self.lambda_val = lambda_val

    def forward(self, multi_layer_latents, inv_perm_A=None, inv_perm_B=None):
        """
        Args:
            multi_layer_latents: List[Tensor], each of shape (2, N, ...)
            inv_perm_A: (B, N)
            inv_perm_B: (B, N)
        Returns:
            loss: scalar
            indices: list of (row_ind, col_ind) per batch
        """
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
            latent_A = latent[0].view(N, -1).to(device)  # shape (N, latent_dim)
            latent_B = latent[1].view(N, -1).to(device)

            # Compute per-layer cost
            cost = compute_distance_matrix(latent_A, latent_B, self.distance_type)
            combined_cost_matrix += weight * cost

            # Hungarian matching and ideal matching matrix
            predicted_matching = LAPSolver.apply(cost, params)
            ideal_matching = torch.zeros_like(predicted_matching)
            ideal_matching[inv_perm_A, inv_perm_B] = 1.0

            # Hamming loss per layer
            loss = HammingLoss()(predicted_matching, ideal_matching)
            total_loss += weight * loss

        final_predicted_matching = LAPSolver.apply(combined_cost_matrix, params)
        col_ind = final_predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(N)

        return total_loss, (row_ind, col_ind)

def build_loss(args):
    """
    Construct the loss function dynamically based on input arguments.
    Args:
        args (dict): Configuration dictionary with keys:
            - USE_MULTI_LAYER_MATCHING: bool
            - LAYER_WEIGHTS: list of float (required if multi-layer)
            - DISTANCE_TYPE: str, one of ["MSE", "L1", "L2"]
            - LAMBDA: float
    Returns:
        torch.nn.Module: the selected loss function
    """
    if args["USE_MULTI_LAYER_MATCHING"]:
        return MultiLayerHungarianLoss(
            layer_weights=args.get("LAYER_WEIGHTS", [0.5, 0.5]),
            distance_type=args.get("DISTANCE_TYPE", "MSE"),
            lambda_val=args.get("LAMBDA", 20)
        )
    else:
        return DifferentiableHungarianLoss(
            distance_type=args.get("DISTANCE_TYPE", "MSE"),
            lambda_val=args.get("LAMBDA", 20)
        )
