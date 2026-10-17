"""NNUE trainer with WDL + search-score blended loss, Adam optimizer, and cosine LR schedule."""

from dataclasses import dataclass
import math
import os
from typing import List, Optional, Tuple
import numpy as np

from .dataset import NNUEDataset
from .model import NNUEGradients, NNUEModel


@dataclass
class TrainingConfig:
    # Hyperparameters for NNUE training.
    epochs: int = 10
    batch_size: int = 128
    lr: float = 0.002
    min_lr: float = 0.0001
    wdl_lambda: float = 0.75
    eval_scale: float = 400.0
    weight_decay: float = 1e-5
    seed: int = 42
    warm_start: bool = True


@dataclass
class EpochMetrics:
    # Per-epoch training metrics.
    epoch: int
    lr: float
    train_loss: float
    val_loss: float


# Load a TrainingConfig from a simple key: value YAML configuration file.
def load_training_config(path: str) -> TrainingConfig:
    if not os.path.isfile(path):
        raise ValueError(f"Training config file not found: {path}")
    cfg = TrainingConfig()
    with open(path, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, val = [p.strip() for p in line.split(":", 1)]
            if key == "epochs":
                cfg.epochs = int(val)
            elif key == "batch_size":
                cfg.batch_size = int(val)
            elif key == "lr":
                cfg.lr = float(val)
            elif key == "min_lr":
                cfg.min_lr = float(val)
            elif key == "wdl_lambda":
                cfg.wdl_lambda = float(val)
            elif key == "eval_scale":
                cfg.eval_scale = float(val)
            elif key == "weight_decay":
                cfg.weight_decay = float(val)
            elif key == "seed":
                cfg.seed = int(val)
            elif key == "warm_start":
                cfg.warm_start = val.lower() in ("true", "1", "yes")
    return cfg


# Numerically stable sigmoid function over a NumPy array.
def sigmoid(x: np.ndarray) -> np.ndarray:
    clamped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clamped))


# Compute cosine annealing learning rate for `epoch` in `[0, total_epochs - 1]`.
def cosine_lr(epoch: int, total_epochs: int, base_lr: float, min_lr: float = 1e-4) -> float:
    if total_epochs <= 1:
        return base_lr
    progress = float(epoch) / float(max(1, total_epochs - 1))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_decay


# Compute MSE loss in WDL probability space blending search eval and game result, plus gradient wrt `pred_cp`.
def wdl_blended_loss(
    pred_cp: np.ndarray,
    target_cp: np.ndarray,
    target_wdl: np.ndarray,
    wdl_lambda: float = 0.75,
    eval_scale: float = 400.0,
) -> Tuple[float, np.ndarray]:
    pred_prob = sigmoid(pred_cp / eval_scale)
    eval_prob = sigmoid(target_cp / eval_scale)
    blended_target = wdl_lambda * eval_prob + (1.0 - wdl_lambda) * target_wdl

    diff = pred_prob - blended_target
    loss = float(np.mean(diff * diff))

    n = max(1, len(pred_cp))
    dsig = (pred_prob * (1.0 - pred_prob)) / eval_scale
    grad_pred_cp = (2.0 / float(n)) * diff * dsig
    return loss, grad_pred_cp.astype(np.float32)


class AdamOptimizer:
    """Adam optimizer with decoupled weight decay for NNUEModel parameters."""

    def __init__(
        self,
        model: NNUEModel,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        self.m_w1 = np.zeros_like(model.w1)
        self.v_w1 = np.zeros_like(model.w1)
        self.m_b1 = np.zeros_like(model.b1)
        self.v_b1 = np.zeros_like(model.b1)
        self.m_w2 = np.zeros_like(model.w2)
        self.v_w2 = np.zeros_like(model.w2)
        self.m_b2 = 0.0
        self.v_b2 = 0.0

    # Apply one Adam parameter update step using `grads` and learning rate `lr`.
    def step(self, model: NNUEModel, grads: NNUEGradients, lr: float) -> None:
        self.step_count += 1
        b1_corr = 1.0 - (self.beta1 ** self.step_count)
        b2_corr = 1.0 - (self.beta2 ** self.step_count)

        # w1 update
        self.m_w1 = self.beta1 * self.m_w1 + (1.0 - self.beta1) * grads.w1
        self.v_w1 = self.beta2 * self.v_w1 + (1.0 - self.beta2) * (grads.w1 * grads.w1)
        m_hat_w1 = self.m_w1 / b1_corr
        v_hat_w1 = self.v_w1 / b2_corr
        model.w1 -= lr * (m_hat_w1 / (np.sqrt(v_hat_w1) + self.eps) + self.weight_decay * model.w1)

        # b1 update
        self.m_b1 = self.beta1 * self.m_b1 + (1.0 - self.beta1) * grads.b1
        self.v_b1 = self.beta2 * self.v_b1 + (1.0 - self.beta2) * (grads.b1 * grads.b1)
        m_hat_b1 = self.m_b1 / b1_corr
        v_hat_b1 = self.v_b1 / b2_corr
        model.b1 -= lr * (m_hat_b1 / (np.sqrt(v_hat_b1) + self.eps))

        # w2 update
        self.m_w2 = self.beta1 * self.m_w2 + (1.0 - self.beta1) * grads.w2
        self.v_w2 = self.beta2 * self.v_w2 + (1.0 - self.beta2) * (grads.w2 * grads.w2)
        m_hat_w2 = self.m_w2 / b1_corr
        v_hat_w2 = self.v_w2 / b2_corr
        model.w2 -= lr * (m_hat_w2 / (np.sqrt(v_hat_w2) + self.eps) + self.weight_decay * model.w2)

        # b2 update
        self.m_b2 = self.beta1 * self.m_b2 + (1.0 - self.beta1) * grads.b2
        self.v_b2 = self.beta2 * self.v_b2 + (1.0 - self.beta2) * (grads.b2 * grads.b2)
        m_hat_b2 = self.m_b2 / b1_corr
        v_hat_b2 = self.v_b2 / b2_corr
        model.b2 -= float(lr * (m_hat_b2 / (math.sqrt(v_hat_b2) + self.eps)))

        model.clamp_weights()


# Evaluate mean WDL-blended loss of `model` over `dataset`.
def evaluate_dataset_loss(
    model: NNUEModel,
    dataset: NNUEDataset,
    config: TrainingConfig,
) -> float:
    if len(dataset) == 0:
        return 0.0
    pred_cp = model.forward(dataset.us_matrix, dataset.them_matrix)
    loss, _ = wdl_blended_loss(
        pred_cp=pred_cp,
        target_cp=dataset.scores_cp,
        target_wdl=dataset.wdl,
        wdl_lambda=config.wdl_lambda,
        eval_scale=config.eval_scale,
    )
    return loss


# Train an NNUEModel on `dataset` using Adam + cosine LR and return `(best_model, metrics_history)`.
def train_nnue(
    dataset: NNUEDataset,
    config: Optional[TrainingConfig] = None,
    init_model: Optional[NNUEModel] = None,
    val_dataset: Optional[NNUEDataset] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[NNUEModel, List[EpochMetrics]]:
    cfg = config if config is not None else TrainingConfig()
    rng = np.random.default_rng(cfg.seed)

    model = (
        init_model.copy()
        if init_model is not None
        else NNUEModel.create(seed=cfg.seed, warm_start=cfg.warm_start)
    )
    optimizer = AdamOptimizer(model, weight_decay=cfg.weight_decay)

    history: List[EpochMetrics] = []
    best_loss = float("inf")
    best_model = model.copy()

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        lr = cosine_lr(epoch, cfg.epochs, cfg.lr, cfg.min_lr)
        batch_losses: List[float] = []

        for batch in dataset.iter_batches(batch_size=cfg.batch_size, shuffle=True, rng=rng):
            pred_cp, cache = model.forward_with_cache(batch.us_features, batch.them_features)
            loss, grad_pred = wdl_blended_loss(
                pred_cp=pred_cp,
                target_cp=batch.scores_cp,
                target_wdl=batch.wdl,
                wdl_lambda=cfg.wdl_lambda,
                eval_scale=cfg.eval_scale,
            )
            grads = model.backward(cache, grad_pred)
            optimizer.step(model, grads, lr)
            batch_losses.append(loss)

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss = (
            evaluate_dataset_loss(model, val_dataset, cfg)
            if (val_dataset is not None and len(val_dataset) > 0)
            else train_loss
        )

        metrics = EpochMetrics(
            epoch=epoch + 1,
            lr=lr,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        history.append(metrics)

        if val_loss <= best_loss:
            best_loss = val_loss
            best_model = model.copy()
            if checkpoint_dir:
                best_model.save_checkpoint(os.path.join(checkpoint_dir, "best.npz"))

        if checkpoint_dir:
            model.save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.npz"))

    return best_model, history

