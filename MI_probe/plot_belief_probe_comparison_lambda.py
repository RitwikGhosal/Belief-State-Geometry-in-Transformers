from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hmm_2 import Mess3
from linear import LinearProbe
from simplex import project_3d_to_simplex2d


def belief_probe_comparison_plot(
    dataset_path,
    probe_path,
    output_path="comparison_lambda_0.75_using_0.25_probe.png",
    lambda_obs=0.75,
):

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    data = torch.load(
        dataset_path,
        map_location="cpu",
    )

    acts = data["acts"].float()
    tokens = data["tokens"]

    print("acts:", acts.shape)
    print("tokens:", tokens.shape)


    # --------------------------------------------------
    # Compute exact Bayesian beliefs
    # --------------------------------------------------

    hmm = Mess3(
        lambda_obs=lambda_obs
    )

    exact_beliefs = hmm.belief_states(
        tokens
    )

    exact_beliefs = exact_beliefs.reshape(
        -1,
        exact_beliefs.shape[-1],
    ).float()

    print(
        "exact beliefs:",
        exact_beliefs.shape,
    )


    # --------------------------------------------------
    # Load trained linear probe
    # --------------------------------------------------

    ckpt = torch.load(
        probe_path,
        map_location="cpu",
    )

    probe = LinearProbe(
        transformer=None,
        d_in=ckpt["d_in"],
        d_out=ckpt["d_out"],
    )

    probe.load_state_dict(
        ckpt["state_dict"]
    )

    probe.eval()


    # --------------------------------------------------
    # Decode beliefs from residual activations
    # --------------------------------------------------

    with torch.no_grad():

        predicted_beliefs = probe(
            acts
        )

    print(
        "predicted beliefs:",
        predicted_beliefs.shape,
    )


    # --------------------------------------------------
    # Project 3D simplex beliefs -> 2D
    # --------------------------------------------------

    exact_2d = project_3d_to_simplex2d(
        exact_beliefs
    )

    predicted_2d = project_3d_to_simplex2d(
        predicted_beliefs
    )


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4),
    )


    axes[0].scatter(
        exact_2d[:, 0],
        exact_2d[:, 1],
        s=0.5,
    )

    axes[0].set_title(
        f"Exact Bayesian beliefs\n"
        f"$\\lambda_{{obs}}={lambda_obs:.2f}$"
    )

    axes[0].set_aspect(
        "equal"
    )


    axes[1].scatter(
        predicted_2d[:, 0],
        predicted_2d[:, 1],
        s=0.5,
    )

    axes[1].set_title(
        f"Probe-decoded beliefs\n"
        f"$\\lambda_{{obs}}={lambda_obs:.2f}$"
    )

    axes[1].set_aspect(
        "equal"
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()


belief_probe_comparison_plot(
    dataset_path="/home/ritwik/karpathy/shai/dataset_lambda_0.75.pt",
    probe_path="/home/ritwik/karpathy/shai/probe_lambda_0.25_best.pt",
    output_path="/home/ritwik/karpathy/shai/comparison_lambda_0.75_using_0.25_probe.png",
    lambda_obs=0.75,
)
