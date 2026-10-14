"""Shared pretraining for the three DEC notebooks.

Retain both halves of each pretrained layer pair, use dropout corruption, and
follow Section 3.2 and the iteration schedule in Section 4.2. The notebooks
declare the architecture and budget; short demonstrations remain adaptations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dec_model import Encoder, FullAutoencoder, _validate_positive_int


def pretrain_autoencoder(
    X: np.ndarray,
    *,
    layer_dims: Sequence[int],
    batch_size: int,
    pretrain_steps: int,
    finetune_steps: int,
    lr: float,
    momentum: float,
    lr_step_size: int,
    lr_gamma: float,
    dropout_rate: float,
    device: torch.device | str,
    momentum_burnin_steps: int = 0,
) -> Encoder:
    """Pretrain an encoder while preserving the learned decoder initialization.

    X contains samples by features. Each shallow autoencoder reconstructs its
    clean input after dropout on the input and hidden representation. The first
    decoder and final encoder have linear outputs; other layers use ReLU.
    Full-autoencoder fine-tuning uses the pretrained weights without dropout.
    Reconstruction diagnostics use the training sample, not held-out data.

    Budgets and learning-rate decay are measured in mini-batch updates. SGD
    uses zero weight decay. Training minimizes half the squared error averaged
    over both samples and features, matching the authors' modified Caffe loss
    (dot / count / 2), not upstream Caffe's sample-only normalization.
    Reported reconstruction MSE uses the same average without the factor 1/2.
    Dropout follows the paper's equations, including the two boundary layers.
    With momentum_burnin_steps > 0, ramp momentum linearly from zero to the
    requested value, restarting for each shallow layer and full fine-tuning.
    This follows the authors' Caffe solver; zero preserves the tutorial setting.
    """
    _validate_positive_int("pretrain_steps", pretrain_steps)
    _validate_positive_int("finetune_steps", finetune_steps)
    _validate_positive_int("lr_step_size", lr_step_size)
    if momentum_burnin_steps != 0:
        _validate_positive_int("momentum_burnin_steps", momentum_burnin_steps)
    device = torch.device(device)
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    current_X = X_tensor
    input_dim = X.shape[1]
    pretrained_encoders: list[nn.Linear] = []
    pretrained_decoders: list[nn.Linear] = []

    print("\nStarting Autoencoder Pretraining")
    print(f"Initial input shape: {X.shape}")
    print(f"Input and hidden dropout probability: {dropout_rate}")

    for layer_index, hidden_dim in enumerate(layer_dims):
        print(f"\nPretraining Layer {layer_index + 1}: {input_dim} -> {hidden_dim}")
        encoder_layer = nn.Linear(input_dim, hidden_dim)
        decoder_layer = nn.Linear(hidden_dim, input_dim)
        for layer in (encoder_layer, decoder_layer):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

        layer_encoder = nn.Sequential(
            encoder_layer,
            nn.Identity() if layer_index == len(layer_dims) - 1 else nn.ReLU(),
        )
        layer_decoder = nn.Sequential(
            decoder_layer,
            nn.Identity() if layer_index == 0 else nn.ReLU(),
        )
        dae = nn.Sequential(
            nn.Dropout(dropout_rate),
            layer_encoder,
            nn.Dropout(dropout_rate),
            layer_decoder,
        ).to(device)
        optimizer = torch.optim.SGD(
            dae.parameters(), lr=lr, momentum=momentum, weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
        loader = DataLoader(
            TensorDataset(current_X), batch_size=batch_size, shuffle=True
        )

        batches = iter(loader)
        for step in range(1, pretrain_steps + 1):
            try:
                (batch,) = next(batches)
            except StopIteration:
                batches = iter(loader)
                (batch,) = next(batches)
            batch = batch.to(device)
            optimizer.zero_grad()
            # The authors' Caffe fork divides by all elements, not just samples.
            loss = 0.5 * F.mse_loss(dae(batch), batch, reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite reconstruction loss in pretraining layer "
                    f"{layer_index + 1}, step {step}. Training stopped; "
                    "discard this partial model and check the optimizer settings."
                )
            loss.backward()
            # Check gradient finiteness without changing finite gradients.
            nn.utils.clip_grad_norm_(
                dae.parameters(), max_norm=float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            if momentum_burnin_steps:
                # Fresh SGD has no history on its first update, so that update
                # is identical to momentum zero. Set momentum for the next one.
                for group in optimizer.param_groups:
                    group["momentum"] = momentum * min(
                        step / momentum_burnin_steps, 1.0
                    )
            scheduler.step()

            if step == 1 or step == pretrain_steps or step % 10000 == 0:
                print(
                    f"   Step {step:6d} | Batch reconstruction objective: "
                    f"{loss.item():.6f}"
                )

        dae.eval()
        with torch.no_grad():
            clean_loader = DataLoader(
                TensorDataset(current_X), batch_size=batch_size, shuffle=False
            )
            current_X = torch.cat([
                layer_encoder(batch.to(device)).cpu() for (batch,) in clean_loader
            ])
        pretrained_encoders.append(encoder_layer)
        pretrained_decoders.append(decoder_layer)
        input_dim = hidden_dim
        print(f"Output shape after Layer {layer_index + 1}: {current_X.shape}")

    encoder = Encoder.from_linear_layers(
        X.shape[1], pretrained_encoders, activate_final=False
    ).to(device)
    model = FullAutoencoder(encoder).to(device)
    # The mirrored decoder must start from the trained shallow decoders.
    for target, source in zip(model.decoder_layers, reversed(pretrained_decoders)):
        target.load_state_dict(source.state_dict())

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )
    loader = DataLoader(
        TensorDataset(X_tensor), batch_size=batch_size, shuffle=True
    )
    print("\nFine-tuning full autoencoder from pretrained encoder/decoder weights")
    batches = iter(loader)
    for step in range(1, finetune_steps + 1):
        try:
            (batch,) = next(batches)
        except StopIteration:
            batches = iter(loader)
            (batch,) = next(batches)
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = 0.5 * F.mse_loss(model(batch), batch, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite reconstruction loss in full-autoencoder fine-tuning, "
                f"step {step}. Training stopped; discard this partial model "
                "and check the optimizer settings."
            )
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf"), error_if_nonfinite=True
        )
        optimizer.step()
        if momentum_burnin_steps:
            for group in optimizer.param_groups:
                group["momentum"] = momentum * min(
                    step / momentum_burnin_steps, 1.0
                )
        scheduler.step()

        if step == 1 or step == finetune_steps or step % 10000 == 0:
            print(
                f"   Step {step:6d} | Batch fine-tune objective: "
                f"{loss.item():.6f}"
            )

    # Evaluate one fixed model so its MSE is comparable to the mean-vector baseline.
    model.eval()
    evaluation_loader = DataLoader(
        TensorDataset(X_tensor), batch_size=batch_size, shuffle=False
    )
    total_loss = 0.0
    with torch.no_grad():
        for (batch,) in evaluation_loader:
            batch = batch.to(device)
            total_loss += F.mse_loss(model(batch), batch).item() * len(batch)
        preview = encoder(X_tensor[:10].to(device)).cpu().numpy()

    baseline_mse = torch.mean((X_tensor - X_tensor.mean(dim=0)) ** 2).item()
    reconstruction_mse = total_loss / len(X_tensor)
    print(f"\nMean-vector reconstruction MSE: {baseline_mse:.6f}")
    print(f"Final autoencoder reconstruction MSE: {reconstruction_mse:.6f}")
    print(f"Baseline minus autoencoder MSE: {baseline_mse - reconstruction_mse:.6f}")
    print("Encoder preview (first 10 samples):")
    print("   Shape:", preview.shape)
    print("   Mean per latent dim:", preview.mean(axis=0))
    print("   Std per latent dim:", preview.std(axis=0))
    return encoder
