from pathlib import Path

import torch

from hmm_2 import Mess3
from transformer import BeliefStateTransformer


# --------------------------------------------------
# Device / experiment settings
# --------------------------------------------------

device = torch.device("cpu")

lambda_obs = 0.75

checkpoint_path = Path(
    "/home/ritwik/karpathy/shai/best_lambda_0.75.pt"
)

batch_size = 128
num_sequences = 10_000
seq_len = 10

resid_stage = "post"
layers = [3]


# --------------------------------------------------
# HMM
# --------------------------------------------------

hmm = Mess3(
    lambda_obs=lambda_obs
)

hmm._t_x = hmm._t_x.to(device)
hmm._t = hmm._t.to(device)
hmm._pi = hmm._pi.to(device)
hmm._joint = hmm._joint.to(device)


# --------------------------------------------------
# Transformer
# --------------------------------------------------

model = BeliefStateTransformer.from_paper_config(
    vocab_size=hmm.vocab_size,
    device=device,
)

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
)

model.load_state_dict(state_dict)
model.eval()


# --------------------------------------------------
# Storage
# --------------------------------------------------

acts_list = []
states_list = []
beliefs_list = []
tokens_list = []


# --------------------------------------------------
# Sampling
# --------------------------------------------------

num_remaining = num_sequences

with torch.no_grad():

    while num_remaining > 0:

        current_batch_size = min(
            batch_size,
            num_remaining,
        )

        # Generate observations from the SAME lambda=0.25 HMM
        tokens, hidden = hmm.generate_batch(
            batch_size=current_batch_size,
            seq_len=seq_len,
        )

        tokens = tokens.to(device)
        hidden = hidden.to(device)

        # Exact Bayesian posterior beliefs
        beliefs = hmm.belief_states(tokens)

        # Transformer residual activations
        _, activations = model.forward_with_residuals(
            tokens,
            resid_stage=resid_stage,
            layers=layers,
        )

        # activations:
        # [layer, batch, pos, d_model]

        acts = activations[0]

        # acts:
        # [batch, pos, 64]

        acts = acts.reshape(
            -1,
            acts.shape[-1],
        ).float().cpu()

        beliefs = beliefs.reshape(
            -1,
            beliefs.shape[-1],
        ).float().cpu()

        # hidden has shape [batch, seq_len + 1]
        # align actual hidden states with the 10 token positions
        states = (
            hidden[:, :-1]
            .reshape(-1)
            .long()
            .cpu()
        )

        acts_list.append(acts)
        beliefs_list.append(beliefs)
        states_list.append(states)
        tokens_list.append(tokens.cpu())

        num_remaining -= current_batch_size


# --------------------------------------------------
# Combine batches
# --------------------------------------------------

acts_all = torch.cat(
    acts_list,
    dim=0,
)

beliefs_all = torch.cat(
    beliefs_list,
    dim=0,
)

states_all = torch.cat(
    states_list,
    dim=0,
)

tokens_all = torch.cat(
    tokens_list,
    dim=0,
)


print("acts:", acts_all.shape)
print("beliefs:", beliefs_all.shape)
print("states:", states_all.shape)
print("tokens:", tokens_all.shape)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_dir = Path(
    "/home/ritwik/karpathy/shai"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    output_dir
    / f"dataset_lambda_{lambda_obs:.2f}.pt"
)

torch.save(
    {
        "acts": acts_all,
        "beliefs": beliefs_all,
        "states": states_all,
        "tokens": tokens_all,

        "lambda_obs": lambda_obs,
        "seq_len": seq_len,
        "resid_stage": resid_stage,
        "layers": layers,
        "num_sequences": num_sequences,
        "checkpoint_path": str(checkpoint_path),
    },
    output_path,
)

print(
    f"saved to {output_path}"
)
