import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from linear import LinearProbe

##----------Load Dataset-----------

data = torch.load("/home/ritwik/karpathy/shai/dataset.pt")

acts = data["acts"].float()
beliefs = data["beliefs"].float()

#print("acts:", acts.shape)
#print("beliefs:", beliefs.shape)

#-------------train_eval_dataset/loader-----------

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


##-----------Probe----------------

probe= LinearProbe(
    transformer = None,
    d_in = acts.shape[1],
    d_out = beliefs.shape[1],
)

optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr = 1e-3,
)  

loss_fn = nn.MSELoss()

##---------Train---------------

epochs = 100

for epoch in range(1, epochs + 1):
    probe.train()
    train_loss = 0.0
    train_count = 0

    for batch_acts, batch_beliefs in train_loader:
        preds = probe(batch_acts)
        loss = loss_fn(preds, batch_beliefs,)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*batch_acts.shape[0]
        train_count += batch_acts.shape[0]

    avg_train_loss = train_loss/train_count

    #--------evaluate--------------------

    probe.eval()

    eval_loss = 0.0
    eval_count = 0

    with torch.no_grad():

        for batch_acts, batch_beliefs in eval_loader:

            preds = probe(batch_acts)
            loss = loss_fn(preds, batch_beliefs,)
            eval_loss += loss.item() * batch_acts.shape[0]
            eval_count += batch_acts.shape[0]

    avg_eval_loss = eval_loss / eval_count

    print(
        f"epoch={epoch} "
        f"train_mse={avg_train_loss:.6f} "
        f"eval_mse={avg_eval_loss:.6f}"
    )    

# --------------------
# Save
# --------------------

torch.save(
    {
        "state_dict": probe.state_dict(),
        "d_in": acts.shape[1],
        "d_out": beliefs.shape[1],
    },
    "probe.pt",
)

print("saved probe.pt")

