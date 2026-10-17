"""Quantize floating-point NNUE models to int16 and export binary .nnue files."""

from typing import Optional
import numpy as np

from ..eval.nnue_ref import (
    NNUE_QA,
    NNUE_QB,
    NNUEWeights,
    save_nnue_file,
    set_active_weights,
)
from .model import NNUEModel


# Quantize a floating-point NNUEModel into int16 NNUEWeights (`QA = 255`, `QB = 64`).
def quantize_model(model: NNUEModel) -> NNUEWeights:
    w1_q = np.clip(np.rint(model.w1 * float(NNUE_QA)), -32768, 32767).astype(np.int16)
    b1_q = np.clip(np.rint(model.b1 * float(NNUE_QA)), -32768, 32767).astype(np.int16)
    w2_q = np.clip(np.rint(model.w2 * float(NNUE_QB)), -32768, 32767).astype(np.int16)
    b2_q = int(np.clip(round(model.b2 * float(NNUE_QA * NNUE_QB)), -32768, 32767))
    return NNUEWeights(w1=w1_q, b1=b1_q, w2=w2_q, b2=b2_q)


# Dequantize int16 NNUEWeights back into a floating-point NNUEModel.
def dequantize_weights(weights: NNUEWeights) -> NNUEModel:
    return NNUEModel.from_quantized(weights)


# Quantize `model`, write it to `output_path` in `.nnue` binary format, and optionally activate it.
def export_nnue(
    model: NNUEModel,
    output_path: Optional[str] = None,
    activate: bool = False,
) -> NNUEWeights:
    weights = quantize_model(model)
    if output_path:
        save_nnue_file(output_path, weights)
    if activate:
        set_active_weights(weights)
    return weights

