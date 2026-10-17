"""Dataset loader converting chess positions into sparse/dense NNUE perspective features."""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple
import numpy as np

from ..core.board import Board
from ..core.types import Color
from ..eval.nnue_ref import NNUE_INPUT_SIZE, extract_features
from .datagen import SelfPlaySample, load_binary_samples, load_jsonl_samples

try:
    import torch
    from torch.utils.data import Dataset as _TorchDatasetBase

    HAS_TORCH = True
except ImportError:
    torch = None
    _TorchDatasetBase = object
    HAS_TORCH = False


@dataclass
class NNUEBatch:
    # Mini-batch of perspective binary features `(B, 768)` and targets `(B,)`.
    us_features: np.ndarray
    them_features: np.ndarray
    scores_cp: np.ndarray
    wdl: np.ndarray


# Convert a single Board into `(us_indices, them_indices)` relative to the side to move.
def board_to_sparse_features(board: Board) -> Tuple[List[int], List[int]]:
    w_feats, b_feats = extract_features(board)
    if board.turn == Color.WHITE:
        return w_feats, b_feats
    return b_feats, w_feats


class NNUEDataset(_TorchDatasetBase):
    """Dataset of chess positions encoded as 768-dimensional perspective feature vectors."""

    def __init__(self, samples: Sequence[SelfPlaySample]):
        self.samples: List[SelfPlaySample] = list(samples)
        n = len(self.samples)
        self.us_matrix = np.zeros((n, NNUE_INPUT_SIZE), dtype=np.float32)
        self.them_matrix = np.zeros((n, NNUE_INPUT_SIZE), dtype=np.float32)
        self.scores_cp = np.zeros(n, dtype=np.float32)
        self.wdl = np.zeros(n, dtype=np.float32)
        self.sparse_indices: List[Tuple[List[int], List[int]]] = []

        for idx, sample in enumerate(self.samples):
            board = Board(sample.fen)
            us_idx, them_idx = board_to_sparse_features(board)
            self.sparse_indices.append((us_idx, them_idx))
            if us_idx:
                self.us_matrix[idx, us_idx] = 1.0
            if them_idx:
                self.them_matrix[idx, them_idx] = 1.0
            self.scores_cp[idx] = float(sample.score_cp)
            self.wdl[idx] = float(sample.wdl)

    @classmethod
    def from_file(cls, path: str) -> "NNUEDataset":
        if path.endswith(".jsonl"):
            samples = load_jsonl_samples(path)
        else:
            samples = load_binary_samples(path)
        return cls(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if HAS_TORCH:
            return (
                torch.from_numpy(self.us_matrix[idx]),
                torch.from_numpy(self.them_matrix[idx]),
                torch.tensor(self.scores_cp[idx], dtype=torch.float32),
                torch.tensor(self.wdl[idx], dtype=torch.float32),
            )
        return (
            self.us_matrix[idx],
            self.them_matrix[idx],
            float(self.scores_cp[idx]),
            float(self.wdl[idx]),
        )

    # Split dataset into `(train_dataset, val_dataset)` according to `val_ratio`.
    def split(self, val_ratio: float = 0.1, seed: int = 42) -> Tuple["NNUEDataset", "NNUEDataset"]:
        n = len(self.samples)
        if n <= 1 or val_ratio <= 0.0:
            return self, NNUEDataset([])
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        val_count = max(1, int(round(n * val_ratio)))
        val_idx = perm[:val_count]
        train_idx = perm[val_count:]
        train_samples = [self.samples[int(i)] for i in train_idx]
        val_samples = [self.samples[int(i)] for i in val_idx]
        return NNUEDataset(train_samples), NNUEDataset(val_samples)

    # Yield `NNUEBatch` slices over the dataset with optional shuffling.
    def iter_batches(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterator[NNUEBatch]:
        n = len(self.samples)
        if n == 0:
            return
        indices = np.arange(n)
        if shuffle:
            gen = rng if rng is not None else np.random.default_rng()
            gen.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield NNUEBatch(
                us_features=self.us_matrix[batch_idx],
                them_features=self.them_matrix[batch_idx],
                scores_cp=self.scores_cp[batch_idx],
                wdl=self.wdl[batch_idx],
            )

