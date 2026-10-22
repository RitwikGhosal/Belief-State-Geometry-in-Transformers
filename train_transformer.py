from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import wandb
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hmm_2 import Mess3
from transformer import BeliefStateTransformer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainConfig:
    device: str
    batch_size: int
    seq_len: int
    epochs: int
    learning_rate: float
    optimizer: str
    adamw_weight_decay: float
    adamw_beta1: float
    adamw_beta2: float
    log_interval: int
    save_interval: int
    seed: int
    save_path: Optional[Path]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train transformer on Mess3 observations.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.01)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 3e-4 if args.optimizer == "adamw" else 0.01

    save_path = Path(args.save_path) if args.save_path is not None else None
    return TrainConfig(
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        epochs=args.epochs,
        learning_rate=learning_rate,
        optimizer=args.optimizer,
        adamw_weight_decay=args.adamw_weight_decay,
        adamw_beta1=args.adamw_beta1,
        adamw_beta2=args.adamw_beta2,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        save_path=save_path,
    )


def next_token_loss_from_logits(
    logits: torch.Tensor, tokens: torch.Tensor, loss_fn: nn.Module
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return loss_fn(
        logits[:, :-1, :].reshape(-1, vocab_size),
        tokens[:, 1:].reshape(-1),
    )


def next_token_accuracy_from_logits(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    preds = logits[:, :-1, :].argmax(dim=-1)
    targets = tokens[:, 1:]
    return (preds == targets).to(torch.float32).mean()


def kl_optimal_next_token_from_logits(
    hmm: Mess3,
    tokens: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:

    opt_probs = hmm.optimal_next_token_probs(tokens)[:, :-1, :]
    logits = logits[:, :-1, :]

    log_model = torch.log_softmax(
        logits,
        dim=-1,
    ).to(dtype=opt_probs.dtype)

    log_opt = torch.log(opt_probs)

    return (
        opt_probs * (log_opt - log_model)
    ).sum(dim=-1).mean().to(torch.float32)


def compute_metrics(
    model: BeliefStateTransformer, hmm: Mess3, tokens: torch.Tensor, loss_fn: nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model.forward_tokens(tokens)
    loss = next_token_loss_from_logits(logits, tokens, loss_fn)
    acc = next_token_accuracy_from_logits(logits, tokens)
    kl = kl_optimal_next_token_from_logits(hmm, tokens, logits)
    return loss, acc, kl



def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)

    device = torch.device(config.device)

    # Experimental condition
    lambda_obs = 0.25

    run_id = time.strftime("%Y%m%d_%H%M%S")

    #checkpoint_dir = (
    #    Path("outputs")
    #    / "transformer"
    #    / f"lambda_{lambda_obs:.2f}_{run_id}"
    #)
    
    checkpoint_dir = (
	Path("/content/drive/MyDrive/BeliefStateCheckpoints")
	/ f"lambda_{lambda_obs:.2f}_{run_id}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build HMM for this observation condition
    hmm = Mess3(lambda_obs=lambda_obs)

    mutual_information = hmm.observation_mutual_information()

    # Move HMM tensors to the training device
    hmm._t_x = hmm._t_x.to(device)
    hmm._t = hmm._t.to(device)
    hmm._pi = hmm._pi.to(device)
    hmm._joint = hmm._joint.to(device)

    # Build fresh transformer
    model = BeliefStateTransformer.from_paper_config(
        vocab_size=hmm.vocab_size,
        device=device,
    )

    model.train()

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(
                config.adamw_beta1,
                config.adamw_beta2,
            ),
            weight_decay=config.adamw_weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
        )

    loss_fn = nn.CrossEntropyLoss()

    wandb.init(
        project="belief-state-transformers",
        name=f"lambda_{lambda_obs:.2f}_{run_id}",
        config={
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "lambda_obs": lambda_obs,
            "mutual_information": mutual_information,
            "adamw_weight_decay": config.adamw_weight_decay,
            "adamw_beta1": config.adamw_beta1,
            "adamw_beta2": config.adamw_beta2,
            "log_interval": config.log_interval,
            "save_interval": config.save_interval,
            "seed": config.seed,
            "model_config": {
                "n_layers": model.cfg.n_layers,
                "d_model": model.cfg.d_model,
                "n_ctx": model.cfg.n_ctx,
                "d_head": model.cfg.d_head,
                "n_heads": model.cfg.n_heads,
                "d_mlp": model.cfg.d_mlp,
                "act_fn": model.cfg.act_fn,
            },
        },
    )

    best_eval_kl = float("inf")

    start_time = time.time()

    for step in range(1, config.epochs + 1):
        tokens, _ = hmm.generate_batch(
            batch_size=config.batch_size,
            seq_len=config.seq_len,
        )

        tokens = tokens.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model.forward_tokens(tokens)

        loss = next_token_loss_from_logits(
            logits,
            tokens,
            loss_fn,
        )

        loss.backward()
        optimizer.step()

        # Evaluation / logging
        if step % config.log_interval == 0:
            model.eval()

            with torch.no_grad():
                train_loss, train_acc, train_kl = compute_metrics(
                    model,
                    hmm,
                    tokens,
                    loss_fn,
                )

                eval_tokens, _ = hmm.generate_batch(
                    batch_size=config.batch_size,
                    seq_len=config.seq_len,
                )

                eval_tokens = eval_tokens.to(device)

                eval_loss, eval_acc, eval_kl = compute_metrics(
                    model,
                    hmm,
                    eval_tokens,
                    loss_fn,
                )

            # Save best model according to Bayes-optimal KL
            if eval_kl.item() < best_eval_kl:
                best_eval_kl = eval_kl.item()

                torch.save(
                    model.state_dict(),
                    checkpoint_dir / "best.pt",
                )

                print(
                    f"Saved new best checkpoint "
                    f"(eval_kl={best_eval_kl:.6f})"
                )

            model.train()

            wandb.log(
                {
                    "train/loss": float(train_loss.item()),
                    "train/acc": float(train_acc.item()),
                    "train/kl_optimal": float(train_kl.item()),
                    "eval/loss": float(eval_loss.item()),
                    "eval/acc": float(eval_acc.item()),
                    "eval/kl_optimal": float(eval_kl.item()),
                    "best/eval_kl": best_eval_kl,
                },
                step=step,
            )

            elapsed_s = time.time() - start_time

            print(
                f"step={step} "
                f"train_loss={float(train_loss.item()):.6f} "
                f"train_acc={float(train_acc.item()):.4f} "
                f"train_kl_optimal={float(train_kl.item()):.6f} "
                f"| eval_loss={float(eval_loss.item()):.6f} "
                f"eval_acc={float(eval_acc.item()):.4f} "
                f"eval_kl_optimal={float(eval_kl.item()):.6f} "
                f"elapsed_s={elapsed_s:.2f}"
            )

        # Periodic checkpoints
        if step % config.save_interval == 0:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"step_{step}.pt",
            )

    # Always save final model
    torch.save(
        model.state_dict(),
        checkpoint_dir / "final.pt",
    )

    # Optional user-specified extra save location
    if config.save_path is not None:
        config.save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            config.save_path,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
