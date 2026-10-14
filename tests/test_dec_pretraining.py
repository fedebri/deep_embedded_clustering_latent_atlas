"""Small synthetic regression checks for shared pretraining."""

import numpy as np
import pytest
import torch

from dec_pretraining import pretrain_autoencoder


def test_finetuning_starts_from_both_pretrained_layer_pairs(monkeypatch):
    """Catch decoder reinitialization at the fine-tuning boundary."""
    torch.manual_seed(42)
    sgd = torch.optim.SGD
    shallow_parameters = []
    checked_finetuning = []

    def inspect_optimizer_parameters(parameters, **kwargs):
        parameters = list(parameters)
        if len(shallow_parameters) < 2:
            # References retain the final weights after each shallow training step.
            shallow_parameters.append(parameters)
        else:
            expected_parameters = [
                *shallow_parameters[0][:2],
                *shallow_parameters[1][:2],
                *shallow_parameters[1][2:],
                *shallow_parameters[0][2:],
            ]
            assert len(parameters) == len(expected_parameters)
            for actual, expected in zip(parameters, expected_parameters):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checked_finetuning.append(True)
        assert kwargs["momentum"] == 0.9
        assert kwargs["weight_decay"] == 0.0
        return sgd(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "SGD", inspect_optimizer_parameters)
    X = np.random.default_rng(42).random((8, 4)).astype(np.float32)
    encoder = pretrain_autoencoder(
        X,
        layer_dims=[3, 2],
        batch_size=4,
        pretrain_steps=1,
        finetune_steps=1,
        lr=1e-3,
        momentum=0.9,
        lr_step_size=20000,
        lr_gamma=0.1,
        dropout_rate=0.2,
        device="cpu",
    )

    assert checked_finetuning == [True]
    assert not encoder.training
    with torch.no_grad():
        embedded = encoder(torch.from_numpy(X))
    assert embedded.shape == (8, 2)
    assert torch.isfinite(embedded).all()


@pytest.mark.parametrize("dropout_rate", [-0.1, 1.1])
def test_invalid_dropout_uses_pytorch_validation(dropout_rate):
    with pytest.raises(ValueError, match="dropout probability"):
        pretrain_autoencoder(
            np.zeros((2, 4), dtype=np.float32),
            layer_dims=[2],
            batch_size=2,
            pretrain_steps=1,
            finetune_steps=1,
            lr=1e-3,
            momentum=0.9,
            lr_step_size=20000,
            lr_gamma=0.1,
            dropout_rate=dropout_rate,
            device="cpu",
        )


def test_momentum_burnin_matches_caffe_updates_and_restarts(monkeypatch):
    """Compare actual parameter updates with the Caffe velocity recurrence."""
    torch.manual_seed(42)
    sgd_step = torch.optim.SGD.step
    histories = {}
    counts = {}

    def compare_update(optimizer, *args, **kwargs):
        index = counts.get(optimizer, 0)
        coefficient = 0.9 * min(index / 2, 1.0)
        velocities = histories.setdefault(optimizer, {})
        expected = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                previous = velocities.get(parameter, torch.zeros_like(parameter))
                velocity = coefficient * previous + group["lr"] * parameter.grad
                velocities[parameter] = velocity.detach().clone()
                expected.append((parameter, parameter.detach().clone() - velocity))
        result = sgd_step(optimizer, *args, **kwargs)
        for actual, reference in expected:
            torch.testing.assert_close(actual, reference)
        counts[optimizer] = index + 1
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", compare_update)
    pretrain_autoencoder(
        np.random.default_rng(42).random((8, 4)).astype(np.float32),
        layer_dims=[3, 2], batch_size=4,
        pretrain_steps=4, finetune_steps=4,
        lr=1e-3, momentum=0.9, momentum_burnin_steps=2,
        lr_step_size=20000, lr_gamma=0.1,
        dropout_rate=0.2, device="cpu",
    )
    # Two shallow layers and full fine-tuning each start a fresh schedule.
    assert list(counts.values()) == [4, 4, 4]


@pytest.mark.parametrize("input_dim", [4, 784])
def test_every_pretraining_phase_uses_half_elementwise_mean_squared_error(
    monkeypatch, input_dim,
):
    """The authors' fork divides by N*d, including each layer's feature width."""
    torch.manual_seed(42)
    mse_loss = torch.nn.functional.mse_loss
    checked_widths = []

    def check_reconstruction_gradient(prediction, target, **kwargs):
        loss = mse_loss(prediction, target, **kwargs)
        if prediction.requires_grad:
            # Check the objective actually used by training, not a standalone
            # formula that could pass while the helper still uses the wrong loss.
            def check_gradient(gradient):
                expected = (prediction.detach() - target) / prediction.numel()
                torch.testing.assert_close(gradient, expected)
                checked_widths.append(prediction.shape[1])

            prediction.register_hook(check_gradient)
        return loss

    monkeypatch.setattr(
        torch.nn.functional, "mse_loss", check_reconstruction_gradient
    )
    pretrain_autoencoder(
        np.random.default_rng(42).random((8, input_dim)).astype(np.float32),
        layer_dims=[3, 2], batch_size=4,
        pretrain_steps=1, finetune_steps=1,
        lr=0.1, momentum=0.9, momentum_burnin_steps=1000,
        lr_step_size=20000, lr_gamma=0.1,
        dropout_rate=0.2, device="cpu",
    )
    assert checked_widths == [input_dim, 3, input_dim]


@pytest.mark.parametrize(
    "bad_loss_call, expected_phase", [(1, "pretraining layer 1"), (2, "fine-tuning")]
)
def test_nonfinite_loss_stops_before_another_optimizer_update(
    monkeypatch, bad_loss_call, expected_phase
):
    """NaNs must fail at their source, before propagating into later stages."""
    torch.manual_seed(42)
    mse_loss = torch.nn.functional.mse_loss
    sgd_step = torch.optim.SGD.step
    loss_calls = []
    updates = []

    def corrupt_one_loss(*args, **kwargs):
        loss = mse_loss(*args, **kwargs)
        loss_calls.append(True)
        return loss * float("nan") if len(loss_calls) == bad_loss_call else loss

    def record_update(optimizer, *args, **kwargs):
        updates.append(True)
        return sgd_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "mse_loss", corrupt_one_loss)
    monkeypatch.setattr(torch.optim.SGD, "step", record_update)
    with pytest.raises(FloatingPointError, match=expected_phase):
        pretrain_autoencoder(
            np.ones((4, 3), dtype=np.float32),
            layer_dims=[2], batch_size=4,
            pretrain_steps=1, finetune_steps=1,
            lr=1e-3, momentum=0.9, momentum_burnin_steps=1000,
            lr_step_size=20000, lr_gamma=0.1,
            dropout_rate=0.2, device="cpu",
        )
    assert len(updates) == bad_loss_call - 1
