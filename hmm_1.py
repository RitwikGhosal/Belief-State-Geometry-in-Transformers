"""
This code is for the data generating process. For the specific case, Mess3 process is used. 
Reference git : daniballcells/belief-state-transformers
Author - Ritwik
Version_1
"""

from __future__ import annotations
from typing import Final, Tuple

import torch
from einops import rearrange

def _stationary_distribution(transition: torch.Tensor) -> torch.Tensor:

    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError(f"Transition matrix must be square, but got shape={tuple(transition.shape)}")

    n: int = int(transition.shape[0])
    evals, evecs = torch.linalg.eig(transition.T.to(torch.float64))
    idx: int = int(torch.argmin(torch.abs(evals - torch.tensor(1.0, dtype = evals.dtype))).item())
    v = evecs[:, idx].real
    v = torch.clamp(v, min= 0.0)
    if float(v.sum().item()) == 0.0:
        raise ValueError("Failed to compute stationary distribution, all-zero eigen vectors")
    v = v/v.sum()
    if v.shape!= (n,):
        raise ValueError(f"stationary distribution has wrong shape: {tuple(v.shape)}")
    return v

class Mess3:
    vocab: Final[tuple[str, str, str]] = ("A", "B", "C")

    def __init__(self) -> None:
        t_a = torch.tensor(
            [
                [0.765, 0.00375, 0.00375],
                [0.0425, 0.0675, 0.00375],
                [0.0425, 0.00375, 0.0675],
            ],
            dtype = torch.float64,
        ) 
        t_b = torch.tensor(
            [
                [0.0675, 0.0425, 0.00375],
                [0.00375, 0.765, 0.00375],
                [0.00375, 0.0425, 0.0675],
            ],
            dtype = torch.float64,
        )
        t_c = torch.tensor(
            [
                [0.0675, 0.00375, 0.0425],
                [0.00375, 0.0675, 0.0425],
                [0.00375, 0.00375, 0.765],
            ],
            dtype = torch.float64,
        )

        self._t_x: torch.Tensor = torch.stack([t_a, t_b, t_c], dim = 0) # torch.stack is different than the normal tensor usage,
                                                                        #because it creates a new dimension , when we use dim = 0 in this case.
        self._t: torch.Tensor = t_a + t_b + t_c

        self._pi: torch.Tensor = _stationary_distribution(self._t)
        self._joint: torch.Tensor = rearrange(self._t_x, "x i j -> i (x j)").contiguous()

    @property
    def num_states(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return 3

    def generate_batch(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, but got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, but got {seq_len}")

        device = self._t_x.device

        #Initial hidden state s0 ~ pi (stationary distribution over hidden states)
        states = torch.multinomial(self._pi.to(device), num_samples=  batch_size, replacement=True).to(
            torch.long
        )

        seq = torch.empty((batch_size, seq_len), dtype = torch.long, device = device)
        hidden = torch.empty((batch_size, seq_len+1), dtype = torch.long, device=device)
        hidden[:, 0] = states

        for t in range(seq_len):
            probs = self._joint.index_select(0, states).to(device) # [batch_size, 9] 
            idx = rearrange(torch.multinomial(probs, num_samples = 1, replacement = True), "b 1 -> b") # select one element out of those 9 and convert [batch_size 1]->[batch_size]
            emission = torch.div(idx, 3, rounding_mode = "floor")
            next_state = idx.remainder(3)
            seq[:, t] = emission.to(torch.long)
            states = next_state.to(torch.long)
            hidden[:, t+1] = states

        return seq, hidden


    def belief_states(self, tokens: torch.Tensor) -> torch.Tensor:

        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape (batch, seq_len), got {tuple(tokens.shape)}")
        batch_size = int(tokens.shape[0])
        seq_len = int(tokens.shape[1])
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        device = tokens.device
        eta = self._pi.to(device=device, dtype = torch.float64).expand(batch_size, 3).clone()
        beliefs = torch.empty((batch_size, seq_len, 3), dtype=torch.float64, device=device)

        for t in range(seq_len):
            x_t = tokens[:, t].to(device)
            t_x = self._t_x.index_select(0, x_t).to(device=device, dtype=torch.float64)
            numer = torch.einsum("bi, bij -> bj", eta, t_x) #probability that we were in state i, then emitted the observed token x, and ended up in state j.
            denom = numer.sum(dim = -1, keepdim= True)
            eta = numer/denom
            beliefs[:, t, :] = eta

        return beliefs
    

    def optimal_next_token_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim!=2:
            raise ValueError(f"tokens must have shape (batch, seq_len), got {tuple(tokens.shape)}")
        seq_len = int(tokens.shape[1])
        if seq_len < 2:
            raise ValueError(f"seq_len must be at least 2 to compute next-token probs, got {seq_len}")

        beliefs = self.belief_states(tokens)
        emit = self._t_x.to(device = tokens.device, dtype = torch.float64).sum(dim=-1)
        probs = torch.einsum("bts,xs->btx", beliefs, emit)
        return probs
    

    def optimal_next_token_probs_from_beliefs(self, beliefs: torch.Tensor) -> torch.Tensor:
        if beliefs.ndim not in (2, 3):
            raise ValueError(
                f"beliefs must have shape (batch, states) or (batch, pos, states), got {tuple(beliefs.shape)}"
            )
        emit = self._t_x.to(device=beliefs.device, dtype=torch.float64).sum(dim=-1)
        if beliefs.ndim == 2:
            return torch.einsum("bs,xs->bx", beliefs.to(dtype=torch.float64), emit)
        return torch.einsum("bts,xs->btx", beliefs.to(dtype=torch.float64), emit)

"""
         
if __name__ == "__main__":
    hmm = Mess3()

    # A B A C
    tokens = torch.tensor([
        [0, 1, 0, 2]
    ], dtype=torch.long)

    print("Tokens:")
    print(tokens)

    beliefs = hmm.belief_states(tokens)

    print("\nBelief states:")
    print(beliefs)

    probs = hmm.optimal_next_token_probs(tokens)

    print("\nOptimal next-token probabilities:")
    print(probs)
"""