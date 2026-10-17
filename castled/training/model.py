"""NNUE 768 -> 256x2 -> 1 SCReLU model with analytical gradients and optional PyTorch module."""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from ..core.board import Board
from ..eval.nnue_ref import (
    NNUE_HIDDEN_SIZE,
    NNUE_INPUT_SIZE,
    NNUE_QA,
    NNUE_QB,
    NNUE_SCALE,
    NNUEWeights,
    build_default_weights,
)
from .dataset import HAS_TORCH, board_to_sparse_features, torch


@dataclass
class NNUEGradients:
    # Parameter gradients matching NNUEModel shapes.
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float


class NNUEModel:
    """Floating-point 768 -> 256x2 -> 1 SCReLU neural network for chess position evaluation."""

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: float = 0.0,
    ):
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = float(b2)

    @classmethod
    def from_quantized(cls, weights: NNUEWeights) -> "NNUEModel":
        w1 = weights.w1.astype(np.float32) / float(NNUE_QA)
        b1 = weights.b1.astype(np.float32) / float(NNUE_QA)
        w2 = weights.w2.astype(np.float32) / float(NNUE_QB)
        b2 = float(weights.b2) / float(NNUE_QA * NNUE_QB)
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    @classmethod
    def create(cls, seed: int = 42, warm_start: bool = True) -> "NNUEModel":
        rng = np.random.default_rng(seed)
        if warm_start:
            base = cls.from_quantized(build_default_weights())
            base.w1 += rng.normal(0.0, 0.015, size=base.w1.shape).astype(np.float32)
            base.b1 += rng.normal(0.0, 0.01, size=base.b1.shape).astype(np.float32)
            base.w2 += rng.normal(0.0, 0.02, size=base.w2.shape).astype(np.float32)
            base.clamp_weights()
            return base

        w1 = rng.normal(0.02, 0.05, size=(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)).astype(np.float32)
        b1 = np.full(NNUE_HIDDEN_SIZE, 0.15, dtype=np.float32)
        w2 = rng.normal(0.0, 0.08, size=(NNUE_HIDDEN_SIZE * 2,)).astype(np.float32)
        return cls(w1=w1, b1=b1, w2=w2, b2=0.0)

    def copy(self) -> "NNUEModel":
        return NNUEModel(
            w1=self.w1.copy(),
            b1=self.b1.copy(),
            w2=self.w2.copy(),
            b2=float(self.b2),
        )

    # Clamp floating-point parameters so int16 quantization stays within safe bounds.
    def clamp_weights(self, max_w1: float = 1.98, max_w2: float = 1.98) -> None:
        np.clip(self.w1, -max_w1, max_w1, out=self.w1)
        np.clip(self.b1, -max_w1, max_w1, out=self.b1)
        np.clip(self.w2, -max_w2, max_w2, out=self.w2)
        self.b2 = float(np.clip(self.b2, -max_w2, max_w2))

    # Forward pass returning predicted centipawns `(B,)` and intermediate cache for backprop.
    def forward_with_cache(
        self,
        us_features: np.ndarray,
        them_features: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        h_us = us_features @ self.w1 + self.b1
        h_them = them_features @ self.w1 + self.b1

        c_us = np.clip(h_us, 0.0, 1.0)
        c_them = np.clip(h_them, 0.0, 1.0)

        act_us = c_us * c_us
        act_them = c_them * c_them

        w2_us = self.w2[:NNUE_HIDDEN_SIZE]
        w2_them = self.w2[NNUE_HIDDEN_SIZE:]

        raw_out = act_us @ w2_us + act_them @ w2_them + self.b2
        pred_cp = raw_out * float(NNUE_SCALE)

        cache = {
            "us_features": us_features,
            "them_features": them_features,
            "h_us": h_us,
            "h_them": h_them,
            "c_us": c_us,
            "c_them": c_them,
            "act_us": act_us,
            "act_them": act_them,
        }
        return pred_cp, cache

    # Run forward inference on a batch of perspective feature vectors `(B, 768)`.
    def forward(self, us_features: np.ndarray, them_features: np.ndarray) -> np.ndarray:
        pred_cp, _ = self.forward_with_cache(us_features, them_features)
        return pred_cp

    # Evaluate a single `Board` instance in centipawns using floating-point weights.
    def evaluate_board(self, board: Board) -> float:
        us_idx, them_idx = board_to_sparse_features(board)
        us_vec = np.zeros((1, NNUE_INPUT_SIZE), dtype=np.float32)
        them_vec = np.zeros((1, NNUE_INPUT_SIZE), dtype=np.float32)
        if us_idx:
            us_vec[0, us_idx] = 1.0
        if them_idx:
            them_vec[0, them_idx] = 1.0
        return float(self.forward(us_vec, them_vec)[0])

    # Compute analytical gradients wrt `w1, b1, w2, b2` given `grad_pred_cp` of shape `(B,)`.
    def backward(self, cache: Dict[str, np.ndarray], grad_pred_cp: np.ndarray) -> NNUEGradients:
        grad_raw = (grad_pred_cp * float(NNUE_SCALE)).astype(np.float32)

        act_us = cache["act_us"]
        act_them = cache["act_them"]
        grad_w2_us = act_us.T @ grad_raw
        grad_w2_them = act_them.T @ grad_raw
        grad_w2 = np.concatenate([grad_w2_us, grad_w2_them], axis=0)
        grad_b2 = float(np.sum(grad_raw))

        w2_us = self.w2[:NNUE_HIDDEN_SIZE]
        w2_them = self.w2[NNUE_HIDDEN_SIZE:]

        grad_act_us = np.outer(grad_raw, w2_us)
        grad_act_them = np.outer(grad_raw, w2_them)

        h_us = cache["h_us"]
        h_them = cache["h_them"]
        c_us = cache["c_us"]
        c_them = cache["c_them"]

        mask_us = ((h_us > 0.0) & (h_us < 1.0)).astype(np.float32)
        mask_them = ((h_them > 0.0) & (h_them < 1.0)).astype(np.float32)

        grad_h_us = grad_act_us * (2.0 * c_us) * mask_us
        grad_h_them = grad_act_them * (2.0 * c_them) * mask_them

        us_features = cache["us_features"]
        them_features = cache["them_features"]

        grad_w1 = (us_features.T @ grad_h_us) + (them_features.T @ grad_h_them)
        grad_b1 = np.sum(grad_h_us + grad_h_them, axis=0)

        return NNUEGradients(
            w1=grad_w1.astype(np.float32),
            b1=grad_b1.astype(np.float32),
            w2=grad_w2.astype(np.float32),
            b2=grad_b2,
        )

    # Save floating-point model weights to a compressed `.npz` checkpoint.
    def save_checkpoint(self, path: str) -> None:
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=np.array([self.b2], dtype=np.float32),
        )

    # Load floating-point model weights from a `.npz` checkpoint.
    @classmethod
    def load_checkpoint(cls, path: str) -> "NNUEModel":
        data = np.load(path)
        return cls(
            w1=data["w1"].astype(np.float32),
            b1=data["b1"].astype(np.float32),
            w2=data["w2"].astype(np.float32),
            b2=float(data["b2"][0]),
        )


if HAS_TORCH:

    class TorchNNUEModel(torch.nn.Module):
        """Optional PyTorch Module mirroring the 768 -> 256x2 -> 1 SCReLU architecture."""

        def __init__(self, init_model: NNUEModel | None = None):
            super().__init__()
            self.fc1 = torch.nn.Linear(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
            self.fc2 = torch.nn.Linear(NNUE_HIDDEN_SIZE * 2, 1)
            if init_model is not None:
                with torch.no_grad():
                    self.fc1.weight.copy_(torch.from_numpy(init_model.w1.T))
                    self.fc1.bias.copy_(torch.from_numpy(init_model.b1))
                    self.fc2.weight.copy_(torch.from_numpy(init_model.w2.reshape(1, -1)))
                    self.fc2.bias.copy_(torch.tensor([init_model.b2], dtype=torch.float32))

        def forward(self, us_features, them_features):
            h_us = torch.clamp(self.fc1(us_features), 0.0, 1.0).pow(2)
            h_them = torch.clamp(self.fc1(them_features), 0.0, 1.0).pow(2)
            concat = torch.cat([h_us, h_them], dim=-1)
            return self.fc2(concat).squeeze(-1) * float(NNUE_SCALE)

        def to_numpy_model(self) -> NNUEModel:
            with torch.no_grad():
                w1 = self.fc1.weight.detach().cpu().numpy().T.copy()
                b1 = self.fc1.bias.detach().cpu().numpy().copy()
                w2 = self.fc2.weight.detach().cpu().numpy().reshape(-1).copy()
                b2 = float(self.fc2.bias.detach().cpu().item())
            return NNUEModel(w1=w1, b1=b1, w2=w2, b2=b2)

