# Calibrated active learning

## Setup

- Finite population:  
  $\{(x_i,y_i):i=1,\ldots,N\}$, where $x_i\in\mathbb{R}^d$ is a $d$-dimensional feature for fixed $d$, $y_i\in\mathbb{R}$ is the label, $N$ is the population size.

- Parameter of interest:  
  $$
  \mu_N = \frac{1}{N}\sum_{i=1}^N y_i.
  $$

- Two samples:
  - $S_1=\{(x_i,y_i):i\in A_1\}$, where $A_1$ is the index set of size $n_1$.
  - $S_2=\{x_i:i\in A_2\}$, where $A_2$ is the index set of size $n_2$.

  - Sample $S_1$ can be treated as a small sample, and it is obtained by simple random sampling or other probability sampling designs. We can assume that the associated sampling weights $\{d_i:i\in A_1\}$ are also available.

  - Sample $S_2$ is a large sample, and it only contains features. It is a **biased** sample.

- Our task is to determine how to label examples in $S_2$ to obtain a valid doubly-robust estimator for $\mu_N$.

- Assume we can fit a regression model $\hat{f}(x)$ and an “uncertainty function” $\hat{g}(x)$ using $S_1$. These two functions are those in classical active learning.

## Proposed method

Basic idea is to use empirical likelihood to adjust the selection bias of $S_2$, and use classical active learning to determine how to label its examples efficiently. The new part is that the selection probabilities for labeling are determined by minimizing a variance term associated with the doubly-robust estimator incorporating both $\hat{g}(x)$ and the estimated empirical likelihoods.

- First, we use empirical likelihood to determine $\{\hat{p}_i:i\in A_2\}$. That is, solve
  $$
  \max \sum_{i\in A_2}\log p_i, \quad \text{s.t.}
  $$
  $$
  \sum_{i\in A_2}p_i=1,\quad \sum_{i\in A_2}p_i x_i=\mu_{x},
  $$
  where
  $$
  \mu_x = \frac{1}{N}\sum_{i=1}^N x_i.
  $$
  Actually, we can safely assume $\mu_x$ is available from census. If not, we can use
  $$
  \hat{\mu}_x = \frac{1}{N}\sum_{i\in A_1} d_i x_i,
  $$
  instead, where $\hat{\mu}_x$ is the Horvitz–Thompson (HT) estimator for $\mu_x$ in survey sampling.

- Based on the estimated $\{\hat{p}_i:i\in A_2\}$, we should determine selection probabilities $\{\hat{\pi}_i:i\in A_2\}$ by minimizing the variance of the following doubly-robust estimator:
  $$
  \hat{\mu} = \frac{1}{N}\sum_{i\in A_2}\left\{\hat{f}(x_i)+\frac{\delta_i}{\hat{\pi}_i}\bigl(y_i-\hat{f}(x_i)\bigr)\right\},
  $$
  where $\delta_i\sim\text{Bernoulli}(\hat{\pi}_i)$ for $i\in A_2$.

- It can be shown that
  $$
  \pi_i \propto \hat{p}_i \,\hat{g}(x_i). \quad (\text{check})
  $$

## Theories

We should develop theories for two scenarios:

- $\mu_x$ is available.
- $\mu_x$ is unavailable, and we use $\hat{\mu}_x$ instead.

For the second case, we should incorporate some classical results from survey sampling.

## Experiments

- (To be specified.)


## Extension to General M-Estimation

We now extend the calibrated active learning framework from mean estimation to
general **M-estimation** problems.

### Target parameter

Let $\ell_\theta(x,y)$ be a convex loss function indexed by
$\theta \in \Theta \subset \mathbb{R}^q$.
The population target parameter is defined as

$$\theta^*=\arg\min_{\theta \in \Theta}L(\theta),\qquad L(\theta)=\frac{1}{N}\sum_{i=1}^N\ell_\theta(x_i, y_i).$$

This formulation includes linear and nonlinear least squares,
generalized linear models, and other convex risk minimization problems.

### Prediction-powered calibrated risk estimator

Using the regression model $\hat f$, empirical likelihood weights $\{\hat p_i : i \in A_2\}$, and labeling probabilities
$\{\hat \pi_i : i \in A_2\}$, we define the prediction-powered, calibrated empirical risk
$$
  \hat L(\theta)=\sum_{i \in A_2}\hat p_i\left\{\ell_\theta\!\bigl(x_i, \hat f(x_i)\bigr)+\frac{\delta_i}{\hat \pi_i}\Bigl(\ell_\theta(x_i, y_i)-\ell_\theta\!\bigl(x_i, \hat f(x_i)\bigr)\Bigr)\right\},
$$
where $\delta_i \sim \mathrm{Bernoulli}(\hat \pi_i)$.

For any fixed $\theta$,

$$
\mathbb{E}\!\left[\hat L(\theta)\mid X,Y\right]=\sum_{i \in A_2}\hat p_i \ell_\theta(x_i, y_i),
$$
so $\hat L(\theta)$ is an unbiased estimator of the calibrated population risk.

The corresponding estimator of $\theta^*$ is
$$
\hat \theta = \arg\min_{\theta \in \Theta} \hat L(\theta).
$$

### Optimal active labeling for M-estimation

Let
$$
\Delta_\theta(x,y) = \nabla_\theta \ell_\theta(x,y) - \nabla_\theta \ell_\theta\!\bigl(x,\hat f(x)\bigr).
$$
denote the score correction induced by the pseudo-label $\hat f(x)$.

Under standard regularity conditions, a first-order expansion yields the asymptotic variance approximation
$$
\mathrm{Var}(\hat \theta) \approx H^{-1} \left( \sum_{i \in A_2} \frac{\hat p_i^2}{\hat \pi_i} \mathbb{E}\!\left[ \Delta_{\theta^*}(X,Y) \Delta_{\theta^*}(X,Y)^\top \mid X=x_i \right] \right) H^{-1},
$$
where $H = \nabla_\theta^2 L(\theta^*)$.

Minimizing a scalar summary (e.g., trace) of this variance subject to a
labeling budget constraint $\sum_{i \in A_2} \hat \pi_i = m$
leads to the optimal sampling rule
$$
\boxed{
\hat \pi_i \propto \hat p_i \, \hat g(x_i),
}
$$
where $\hat g(x)$ is a scalar proxy for the conditional magnitude of
the score correction, such as
$$
\hat g(x) = \mathbb{E}\!\left[ \bigl|Y-\hat f(X)\bigr| \mid X=x \right].
$$

This recovers the same calibrated active learning rule as in the mean estimation case.