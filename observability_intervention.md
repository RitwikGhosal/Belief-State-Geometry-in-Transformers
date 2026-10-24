# Observability Intervention: What Shapes a Transformer's Belief Representation?

The original Mess3 experiment gives a surprisingly clean result: a transformer trained only to predict the next token develops internal representations from which the Bayesian belief state of the underlying hidden process can be linearly decoded.

That immediately raises another question.

If the transformer is really representing something belief-like, **what determines the shape of that representation?**

Is it mainly learning the dynamics of the hidden world itself? Or is its internal representation also shaped by how much of that hidden world is actually observable?

This experiment tries to separate those two things.

The basic idea is simple: **keep the hidden-state dynamics exactly the same, and change only how informative the observations are.**

---

## The experiment

Mess3 has three hidden states. Their dynamics are governed by the transition matrix

```math
T_{ij}=P(H_{t+1}=j\mid H_t=i)
```

with

```math
T=
\begin{bmatrix}
0.90 & 0.05 & 0.05 \\
0.05 & 0.90 & 0.05 \\
0.05 & 0.05 & 0.90
\end{bmatrix}
```

This matrix is kept fixed throughout the entire experiment.

So the hidden process - the way the underlying world moves from one latent state to another, never changes.

What changes is the observation channel.

The original Mess3 observation matrix is

```math
E_0=
\begin{bmatrix}
0.85 & 0.075 & 0.075 \\
0.075 & 0.85 & 0.075 \\
0.075 & 0.075 & 0.85
\end{bmatrix}
```

This means that each hidden state usually produces its characteristic observation, but occasionally produces one of the others.

At the other extreme, consider a completely uninformative observation channel:

```math
U=
\begin{bmatrix}
1/3 & 1/3 & 1/3 \\
1/3 & 1/3 & 1/3 \\
1/3 & 1/3 & 1/3
\end{bmatrix}
```

Here, seeing an observation tells us nothing about which hidden state generated it.

I interpolate between these two channels using

```math
E_{\lambda}=(1-\lambda)E_0+\lambda U
```

When $\lambda=0$, we recover the original Mess3 process.

As $\lambda$ increases, the hidden dynamics remain unchanged, but the observations become less informative.

At $\lambda=1$, the observations carry no information at all about the hidden state.

This gives a controlled way of changing **observability** without changing the latent dynamics.

---

## What exactly is the Bayesian belief?

The actual HMM is always in one definite hidden state. But the observer does not see that state directly. It only sees the emitted symbols.

After observing a history such as

```text
A B A C B ...
```

an ideal Bayesian observer can ask:

> Given everything I have seen so far, how likely is each hidden state?

That uncertainty is represented by a belief vector

```math
b_t(i)=P(H_t=i\mid X_{< t})
```

For example,

```math
b_t=
\begin{bmatrix}
0.70 \\
0.20 \\
0.10
\end{bmatrix}
```

means that, given the observation history, the Bayesian observer assigns 70% probability to the first hidden state, 20% to the second, and 10% to the third.

Because the three probabilities sum to one, every valid belief lies on a two-dimensional simplex - a triangle.

This distinction is important:

- the **hidden state** is the actual state of the process;
- the **belief state** is the observer's uncertainty about that hidden state.

The exact posterior can be calculated because, as experimenters, we know the HMM.

In the symbol-labelled transition formulation used here,

```math
T^{(x)}_{ij}
=
P(X_t=x,H_{t+1}=j\mid H_t=i)
```

and under the observation convention used in the code,

```math
T^{(x)}_{ij}
=
T_{ij}E_{\lambda}(x\mid j)
```

Given a current belief $b_t$ and a newly observed symbol $x$, the next Bayesian belief is

```math
b_{t+1}(j)
=
\frac{
\sum_i b_t(i)T_{ij}E_{\lambda}(x\mid j)
}{
\sum_k\sum_i b_t(i)T_{ik}E_{\lambda}(x\mid k)
}
```

This equation is the key to the whole experiment.

Even though $T$ is unchanged, the posterior update depends on the observation matrix $E_{\lambda}$.

So changing observability changes the way an ideal observer updates its beliefs.

Over many different observation histories, these posterior beliefs trace out a characteristic geometry inside the simplex.

That is the geometry shown in the **left-hand panel** of the plots below.

---

## Changing observability changes the belief geometry

One useful way to quantify the observation channel is with the mutual information between the hidden state and the observation:

```math
I(H;X)
=
\sum_{h,x}
P(h,x)
\log
\frac{P(h,x)}
{P(h)P(x)}
```

Using the stationary distribution of the fixed transition matrix, the observation intervention gives approximately:

| $\lambda$ | $I(H;X)$ in nats |
|---:|---:|
| 0.00 | 0.572 |
| 0.25 | 0.313 |
| 0.50 | 0.139 |
| 0.75 | 0.036 |
| 1.00 | 0 |

So increasing $\lambda$ progressively removes information about the hidden state from the observations.

The important point is that this does **not** change the latent transition matrix.

It changes only what can be inferred about that latent process.

At λ = 0.25, the exact Bayesian geometry already changes substantially from the original Mess3 geometry. The sparse, highly structured belief-state attractor becomes much denser.

A transformer was then trained from scratch only on next-token prediction under this modified observation channel.

It never sees the hidden states, the transition matrix, the emission matrix, or the Bayesian posteriors.

After training, I extract the final-layer residual activation

```math
h_t\in\mathbb{R}^{64}
```

at each token position and train a linear probe

```math
\hat b_t=Wh_t+c
```

to predict the exact Bayesian posterior $b_t$.

The result is:

![Exact and probe-decoded belief geometry for lambda 0.25](../plots/comparison_lambda_0.25.png)

At $\lambda=0.25$, the exact Bayesian posterior geometry changes substantially from the original Mess3 geometry. A linear probe trained on the transformer's residual stream recovers a closely corresponding structure.

The left panel is therefore **not** a transformer representation. It is the exact posterior calculated directly from the known HMM.

The right panel is the posterior decoded from the transformer's 64-dimensional residual activations using a linear probe.

The transformer was never explicitly trained to reproduce the left-hand geometry.

It was trained only to predict the next observation.

---

The effect becomes even clearer at $\lambda=0.75$.

At this point,

```math
I(H;X)\approx0.036\text{ nats}
```

so the observations contain very little information about the hidden state.

The underlying hidden-state transition matrix is still exactly the same, but the Bayesian observer can no longer become as confident about the latent state.

The reachable belief geometry therefore contracts into a much smaller region of the simplex.

Again, a separately trained transformer develops residual activations from which the corresponding geometry can be linearly decoded:

![Exact and probe-decoded belief geometry for lambda 0.75](../plots/comparison_lambda_0.75.png)

At $\lambda=0.75$, observations are only weakly informative. The exact Bayesian belief geometry contracts strongly, and a probe trained on the corresponding transformer recovers a very similar geometry.

So across the experiments, the pattern is:

```math
\text{same latent dynamics}
+
\text{different observability}
```

```math
\Downarrow
```

```math
\text{different Bayesian inference geometry}
```

```math
\Downarrow
```

```math
\text{different linearly decodable transformer representation}
```

The current evidence therefore suggests that the transformer's internal predictive state is not determined by the hidden transition dynamics alone.

It appears to be shaped by the **inference problem**: what the model can know about the hidden process from the observations it receives.

I do not interpret this as the transformer possessing an explicit abstract concept of “observability”.

A more conservative interpretation is:

> **The transformer's internal predictive representation systematically adapts to how informative the observations are about the underlying latent state.**

---

## Does the representation itself reorganize?

There is an important caveat to the previous plots.

A different linear probe was trained for each transformer.

So the successful $\lambda=0.25$ and $\lambda=0.75$ plots show that belief states are linearly decodable in both models, but they do not by themselves show that both models use the same internal coordinate system.

To test this, I took the probe trained on the $\lambda=0.25$ transformer and applied it directly to residual activations from the $\lambda=0.75$ transformer.

In other words, instead of using

```math
W_{0.75}h_t^{(0.75)}
```

I evaluated

```math
W_{0.25}h_t^{(0.75)}
```

The result looks very different:

![Lambda 0.75 residuals decoded using the lambda 0.25 probe](../plots/comparison_lambda_0.75_using_0.25_probe.png)

The left panel is still the correct Bayesian geometry for $\lambda=0.75$. The right panel uses the probe learned at $\lambda=0.25$ to decode the $\lambda=0.75$ transformer's residual stream.

The cross-decoded representation is not random — it remains highly structured — but it no longer matches the correct Bayesian geometry.

This is an interesting preliminary result.

Within each condition,

```math
W_{0.25}h_t^{(0.25)}
\approx
b_t^{(0.25)}
```

and

```math
W_{0.75}h_t^{(0.75)}
\approx
b_t^{(0.75)}
```

but

```math
W_{0.25}h_t^{(0.75)}
\not\approx
b_t^{(0.75)}
```

This suggests that changing observability may do more than simply move the model through different belief states inside one fixed neural coordinate system.

The way belief information is embedded in residual space may itself reorganize.

That claim needs more careful quantitative testing, including the reverse cross-probe direction and multiple random seeds, so I treat it as a preliminary observation rather than a conclusion.

---

## What I think this is telling us

The original belief-state result shows that a transformer trained for prediction can develop an internal representation from which Bayesian beliefs about a hidden process are linearly accessible.

The intervention here asks a slightly different question:

> What determines the form of that internal belief representation?

The hidden dynamics in these experiments never change.

What changes is the relationship between the hidden world and the evidence available to the model.

That changes what an ideal observer should believe after seeing a particular history, and it changes the geometry of all such possible beliefs.

The transformer appears to follow those changes.

A useful way to think about the computation is therefore

```math
\text{observation history}
\longrightarrow
\text{internal predictive state}
\longrightarrow
\text{next-token prediction}
```

For this HMM, the exact Bayesian posterior

```math
P(H_t\mid X_{1:t-1})
```

is a compact summary of the history that contains the information required for optimal future prediction.

The transformer is never told to construct this posterior.

Nevertheless, next-token training produces an internal state from which it can be linearly recovered.

And when the observations reveal less about the hidden world, that internal state changes along with the corresponding Bayesian inference problem.

The strongest claim supported by the current experiments is therefore not that the transformer has “learned observability” as an abstract concept.

It is that:

> **The learned predictive representation appears to encode not only structure in the latent process, but also uncertainty about that process induced by partial observability.**

This is still an early result. The current evidence is mainly geometric and comes from individual training runs. The next steps are to repeat the experiments across seeds, quantify the geometry and cross-probe failures, and construct observation channels with similar mutual information but different state-confusion structure.

That last experiment is particularly important: it would help distinguish whether the learned representation depends mainly on **how much** information is available, or on **which distinctions about the hidden world** the observations preserve.
