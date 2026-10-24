import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from linear import LinearProbe


# --------------------------------------------------
# Experiment setting
# --------------------------------------------------

lambda_obs = 0.75

torch.manual_seed(0)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset_path = f"dataset_lambda_{lambda_obs:.2f}.pt"

data = torch.load(
    dataset_path,
    map_location="cpu",
)

acts = data["acts"].float()
beliefs = data["beliefs"].float()

print("acts shape:", acts.shape)
print("beliefs shape:", beliefs.shape)


# --------------------------------------------------
# Train / eval split
# --------------------------------------------------

num_samples = acts.shape[0]
num_train = int(0.8 * num_samples)

perm = torch.randperm(num_samples)

train_idx = perm[:num_train]
eval_idx = perm[num_train:]


train_dataset = TensorDataset(
    acts[train_idx],
    beliefs[train_idx],
)

eval_dataset = TensorDataset(
    acts[eval_idx],
    beliefs[eval_idx],
)


train_loader = DataLoader(
    train_dataset,
    batch_size=4096,
    shuffle=True,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=4096,
    shuffle=False,
)


# --------------------------------------------------
# Probe
# --------------------------------------------------

probe = LinearProbe(
    transformer=None,
    d_in=acts.shape[1],
    d_out=beliefs.shape[1],
)


optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=1e-2,
)

loss_fn = nn.MSELoss()


# --------------------------------------------------
# Train
# --------------------------------------------------

epochs = 150

best_eval_loss = float("inf")


for epoch in range(1, epochs + 1):

    probe.train()

    train_loss = 0.0
    train_count = 0

    for batch_acts, batch_beliefs in train_loader:

        preds = probe(batch_acts)

        loss = loss_fn(
            preds,
            batch_beliefs,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += (
            loss.item()
            * batch_acts.shape[0]
        )

        train_count += batch_acts.shape[0]

    avg_train_loss = (
        train_loss
        / train_count
    )


    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    probe.eval()

    eval_loss = 0.0
    eval_count = 0

    with torch.no_grad():

        for batch_acts, batch_beliefs in eval_loader:

            preds = probe(batch_acts)

            loss = loss_fn(
                preds,
                batch_beliefs,
            )

            eval_loss += (
                loss.item()
                * batch_acts.shape[0]
            )

            eval_count += batch_acts.shape[0]


    avg_eval_loss = (
        eval_loss
        / eval_count
    )


    print(
        f"epoch={epoch} "
        f"train_mse={avg_train_loss:.6f} "
        f"eval_mse={avg_eval_loss:.6f}"
    )


    # --------------------------------------------------
    # Save best probe
    # --------------------------------------------------

    if avg_eval_loss < best_eval_loss:

        best_eval_loss = avg_eval_loss

        torch.save(
            {
                "state_dict": probe.state_dict(),
                "d_in": acts.shape[1],
                "d_out": beliefs.shape[1],
                "lambda_obs": lambda_obs,
                "eval_mse": best_eval_loss,
            },
            f"probe_lambda_{lambda_obs:.2f}_best.pt",
        )


# --------------------------------------------------
# Save final probe
# --------------------------------------------------

torch.save(
    {
        "state_dict": probe.state_dict(),
        "d_in": acts.shape[1],
        "d_out": beliefs.shape[1],
        "lambda_obs": lambda_obs,
        "eval_mse": avg_eval_loss,
    },
    f"probe_lambda_{lambda_obs:.2f}_final.pt",
)


print(
    f"saved "
    f"probe_lambda_{lambda_obs:.2f}_best.pt "
    f"and "
    f"probe_lambda_{lambda_obs:.2f}_final.pt"
)
