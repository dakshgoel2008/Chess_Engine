"""Tests for Phase 12: NNUE training pipeline (datagen, dataset, model backprop, trainer, export)."""

import os
import tempfile
import numpy as np
import pytest

from castled.core.board import Board
from castled.core.types import STARTING_FEN
from castled.eval.nnue_ref import evaluate_nnue, reset_default_weights
from castled.training import (
    NNUEDataset,
    NNUEModel,
    SelfPlaySample,
    TrainingConfig,
    board_to_sparse_features,
    cosine_lr,
    dequantize_weights,
    evaluate_dataset_loss,
    export_nnue,
    generate_dataset,
    load_binary_samples,
    load_jsonl_samples,
    load_training_config,
    quantize_model,
    save_binary_samples,
    save_jsonl_samples,
    train_nnue,
    wdl_blended_loss,
)


@pytest.fixture(autouse=True)
def _restore_nnue_after_test():
    reset_default_weights()
    yield
    reset_default_weights()


def test_datagen_and_binary_jsonl_serialization():
    samples = generate_dataset(num_games=2, random_plies=4, search_depth=1, max_plies=24, seed=101)
    assert len(samples) > 0
    for s in samples:
        assert s.wdl in (0.0, 0.5, 1.0)
        assert -30000 <= s.score_cp <= 30000

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "samples.bin")
        n_bin = save_binary_samples(bin_path, samples)
        assert n_bin == len(samples)
        loaded_bin = load_binary_samples(bin_path)
        assert len(loaded_bin) == len(samples)
        for orig, rec in zip(samples, loaded_bin):
            assert orig.score_cp == rec.score_cp
            assert orig.wdl == pytest.approx(rec.wdl)
            assert orig.turn == rec.turn
            b_orig = Board(orig.fen)
            b_rec = Board(rec.fen)
            for sq in range(64):
                assert b_orig.piece_at(sq) == b_rec.piece_at(sq)

        jsonl_path = os.path.join(tmpdir, "samples.jsonl")
        save_jsonl_samples(jsonl_path, samples)
        loaded_jsonl = load_jsonl_samples(jsonl_path)
        assert len(loaded_jsonl) == len(samples)
        assert loaded_jsonl[0].fen == samples[0].fen

        bad_bin = os.path.join(tmpdir, "bad.bin")
        with open(bad_bin, "wb") as fp:
            fp.write(b"NOPE0000")
        with pytest.raises(ValueError):
            load_binary_samples(bad_bin)


def test_dataset_features_split_and_batches():
    samples = [
        SelfPlaySample(fen=STARTING_FEN, score_cp=25, wdl=0.5, turn=0),
        SelfPlaySample(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            score_cp=-25,
            wdl=0.5,
            turn=1,
        ),
        SelfPlaySample(
            fen="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            score_cp=110,
            wdl=1.0,
            turn=0,
        ),
        SelfPlaySample(
            fen="8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            score_cp=-80,
            wdl=0.0,
            turn=0,
        ),
    ]
    ds = NNUEDataset(samples)
    assert len(ds) == 4
    assert ds.us_matrix.shape == (4, 768)
    assert ds.them_matrix.shape == (4, 768)
    assert int(np.sum(ds.us_matrix[0])) == 32
    assert int(np.sum(ds.them_matrix[0])) == 32

    # Verify perspective flip for Black to move
    b_black = Board(samples[1].fen)
    us_idx, them_idx = board_to_sparse_features(b_black)
    assert len(us_idx) == 32 and len(them_idx) == 32
    assert us_idx != them_idx

    train_ds, val_ds = ds.split(val_ratio=0.25, seed=7)
    assert len(train_ds) == 3
    assert len(val_ds) == 1

    batches = list(ds.iter_batches(batch_size=2, shuffle=False))
    assert len(batches) == 2
    assert batches[0].us_features.shape == (2, 768)


def test_model_analytical_gradients_match_finite_differences():
    model = NNUEModel.create(seed=12, warm_start=False)
    ds = NNUEDataset(
        [
            SelfPlaySample(fen=STARTING_FEN, score_cp=40, wdl=1.0, turn=0),
            SelfPlaySample(
                fen="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
                score_cp=-60,
                wdl=0.0,
                turn=1,
            ),
        ]
    )

    pred_cp, cache = model.forward_with_cache(ds.us_matrix, ds.them_matrix)
    _, grad_pred = wdl_blended_loss(pred_cp, ds.scores_cp, ds.wdl, wdl_lambda=0.75, eval_scale=400.0)
    grads = model.backward(cache, grad_pred)

    eps = 1e-3

    # Check b2 finite difference
    orig_b2 = model.b2
    model.b2 = orig_b2 + eps
    loss_plus, _ = wdl_blended_loss(model.forward(ds.us_matrix, ds.them_matrix), ds.scores_cp, ds.wdl)
    model.b2 = orig_b2 - eps
    loss_minus, _ = wdl_blended_loss(model.forward(ds.us_matrix, ds.them_matrix), ds.scores_cp, ds.wdl)
    model.b2 = orig_b2
    num_grad_b2 = (loss_plus - loss_minus) / (2.0 * eps)
    assert grads.b2 == pytest.approx(num_grad_b2, rel=2e-2, abs=1e-4)

    # Check w2[0] finite difference
    orig_w2_0 = float(model.w2[0])
    model.w2[0] = orig_w2_0 + eps
    loss_plus, _ = wdl_blended_loss(model.forward(ds.us_matrix, ds.them_matrix), ds.scores_cp, ds.wdl)
    model.w2[0] = orig_w2_0 - eps
    loss_minus, _ = wdl_blended_loss(model.forward(ds.us_matrix, ds.them_matrix), ds.scores_cp, ds.wdl)
    model.w2[0] = orig_w2_0
    num_grad_w2_0 = (loss_plus - loss_minus) / (2.0 * eps)
    assert float(grads.w2[0]) == pytest.approx(num_grad_w2_0, rel=2e-2, abs=1e-4)


def test_training_convergence_checkpoint_and_export_quantization():
    cfg_path = os.path.join("castled", "training", "configs", "default.yaml")
    loaded_cfg = load_training_config(cfg_path)
    assert loaded_cfg.epochs == 12
    assert loaded_cfg.wdl_lambda == pytest.approx(0.75)

    assert cosine_lr(0, 10, 0.01, 0.001) == pytest.approx(0.01)
    assert cosine_lr(9, 10, 0.01, 0.001) == pytest.approx(0.001)

    samples = [
        SelfPlaySample(fen=STARTING_FEN, score_cp=20, wdl=0.5, turn=0),
        SelfPlaySample(
            fen="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            score_cp=180,
            wdl=1.0,
            turn=0,
        ),
        SelfPlaySample(
            fen="8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            score_cp=-150,
            wdl=0.0,
            turn=0,
        ),
        SelfPlaySample(
            fen="rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            score_cp=260,
            wdl=1.0,
            turn=0,
        ),
    ] * 8

    ds = NNUEDataset(samples)
    cfg = TrainingConfig(epochs=8, batch_size=8, lr=0.004, min_lr=0.0005, wdl_lambda=0.8, seed=99)
    init_model = NNUEModel.create(seed=99, warm_start=True)
    loss_before = evaluate_dataset_loss(init_model, ds, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpts")
        trained_model, history = train_nnue(ds, cfg, init_model=init_model, checkpoint_dir=ckpt_dir)
        loss_after = evaluate_dataset_loss(trained_model, ds, cfg)

        assert len(history) == 8
        assert loss_after < loss_before
        assert os.path.isfile(os.path.join(ckpt_dir, "best.npz"))

        reloaded = NNUEModel.load_checkpoint(os.path.join(ckpt_dir, "best.npz"))
        np.testing.assert_allclose(reloaded.w1, trained_model.w1)

        nnue_path = os.path.join(tmpdir, "trained.nnue")
        q_weights = export_nnue(trained_model, nnue_path, activate=True)
        assert os.path.isfile(nnue_path)

        deq_model = dequantize_weights(quantize_model(trained_model))
        for sample in samples[:4]:
            b = Board(sample.fen)
            c_eval = evaluate_nnue(b)
            float_eval = deq_model.evaluate_board(b)
            # Quantized C int16 forward pass closely tracks float32 forward pass.
            assert abs(c_eval - float_eval) <= 10.0
            assert q_weights.w1.dtype == np.int16

