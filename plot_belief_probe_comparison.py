from pathlib import Path
import matplotlib.pyplot as plt
import torch
from hmm_1 import Mess3
from linear import LinearProbe
from simplex import project_3d_to_simplex2d

def belief_probe_comparison_plot(dataset_path, probe_path, output_path="comparison.png",):

    #----------load dataset---------
    data = torch.load(dataset_path, map_location = "cpu",)
    acts = data['acts'].float()
    tokens = data['tokens']

    #----------compute exact beliefs----------
    hmm = Mess3()
    optimal_beliefs = hmm.belief_states(tokens)
    optimal_beliefs = optimal_beliefs.reshape(-1, optimal_beliefs.shape[-1],).float()

    #------load trained probe----------
    ckpt = torch.load(probe_path, map_location = "cpu",)
    probe = LinearProbe(transformer=None, d_in=ckpt["d_in"], d_out=ckpt["d_out"],)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    #------predict beliefs------------

    with torch.no_grad():

        predicted_beliefs = probe(
            acts
        )


    # -----------------------
    # Project 3D beliefs → 2D
    # -----------------------

    optimal_2d = project_3d_to_simplex2d(
        optimal_beliefs
    )

    predicted_2d = project_3d_to_simplex2d(
        predicted_beliefs
    )


    # -----------------------
    # Plot
    # -----------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4),
    )

    axes[0].scatter(
        optimal_2d[:, 0],
        optimal_2d[:, 1],
        s=0.5,
    )

    axes[0].set_title(
        "True HMM beliefs"
    )

    axes[0].set_aspect("equal")


    axes[1].scatter(
        predicted_2d[:, 0],
        predicted_2d[:, 1],
        s=0.5,
    )

    axes[1].set_title(
        "Probe-predicted beliefs"
    )

    axes[1].set_aspect("equal")


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

belief_probe_comparison_plot(
    dataset_path="/home/ritwik/karpathy/shai/dataset.pt",
    probe_path="/home/ritwik/karpathy/shai/probe.pt",
    output_path="/home/ritwik/karpathy/shai/comparison.png",
)