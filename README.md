# **proximal_policy_optimization_PPO**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Repo size](https://img.shields.io/github/repo-size/Mhddaraaa/proximal_policy_optimization_PPO)
![Last Commit](https://img.shields.io/github/last-commit/Mhddaraaa/proximal_policy_optimization_PPO)

# __Table of content__

<ul>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#vanilla-policy-gradient-vpg">Vanilla policy gradient (VPG)</a></li>
  <li><a href="#kullback-liebler-divergence-kl-divergence">Kullback-Liebler divergence (KL divergence)</a>
    <ul>
      <li><a href="#example-for-discrete-distribution">Example for discrete distribution</a></li>
      <li><a href="#how-we-get-there">How we get there</a></li>
      <li><a href="#add-the-new-term-to-derivarive">Add the new term to derivarive</a></li>
    </ul>
  </li>
  <li><a href="#clipping-objective">Clipping objective</a></li>
  <li><a href="#dynamic-kl-penalty">Dynamic KL Penalty</a></li>
  <li><a href="#model-output">Model output</a></li>
</ul>


## Usage

**install dependencies**
```sh
pip install -r requirements.txt
```

**Run a discrete env with pretrained models**

```python
python main.py -n CartPole-v1 -p './pretrain_models/' -e 500 -m kl
```
```python
python main.py -n LunarLanderContinuous-v3 -p './pretrain_models/' -e 500 -c
```
> "-n" or "--env-name": Gym environment name, like "CartPole-v1", "LunarLander-v3", and "LunarLanderContinuous-v3" <br>
> "-p" or "--model-path: Policy and value models path<br>
> "-e" or  "--num-episode": Number of episodes, even it test mode it should be mention, default=500<br>
> "-l" or "--learning-rate": Neural networks learning rate, default=0.001<br>
> "-nh" or "--num-hidden": Number of hidden layer in neural networks, defaul=64<br>
> "-o or "--optimizer-step": After each 50 episodes learning rate reduce by this factor, defaulr=0.99<br>
> "-c" or "--continuous": If the environment is continuous we should add this flag<br>
> "-t" or "--test-agent": Test the trained agent and record it's palying, with -e flag specify how many episodes it should play<br>
> "-b" or "--bootstrapping": If we want to calculate return reward using bootstrapping, we should add this flag<br>
> "-u" or "--update-steps": Number of times to update the networks in each episode<br>
> "-m" or "--update-mode": Using KL penaly or clipping objective, default='clipping', another option is -m kl<br>
<br>


## Vanilla policy gradient (VPG)

We have been using this method until now, which is a stochastic policy. It explore the environment without using $\epsilon-greedy$.

$$
\large \theta = \theta + \alpha \nabla_\theta J(\theta)|_{\theta = \theta_o}
$$
> $\theta_o$: means $\theta_{old}$

<br>

The gradient evaluate old policy based on $\theta$ parameter and update it based on learning rate ($\alpha$). Nonetheless, A tiny change in $\theta$ could cause a significant change in the policy probabilities. In fact, we would like to keep the old and new policies close to each other in $\pi_\theta(s_t, a_t)$ and not in $\theta$.

<br>

## Kullback-Liebler divergence (KL divergence)

it is a measure of how one probability distribution $Q$ (*the approximate distribution*) differs from a second, reference probability distribution $P$ (*the true distribution*).

Discrete probability distributions:

$$
D_{\text{KL}}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
$$

<br>

Continuous distributions:

$$
\begin{split}
D_{\text{KL}}(P \| Q) & = \int_{-\infty}^{\infty} P(x) \log \frac{P(x)}{Q(x)} \, dx \\
\\
\text{Gaussian distributions: }\\
\\
\text{KL}(P || Q) & = \log\left(\frac{\sigma_{\text{old}}}{\sigma}\right) + \frac{\sigma^2 + (\mu - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2} - 0.5
\end{split}
$$

> $P(x)$ : The true probability distribution (e.g., target distribution). <br>$Q(x)$ : The approximate probability distribution. <br>$\mathcal{X}$ : The support (range of possible values) of the distributions. <br>$\log$ : Typically the natural logarithm.

---
- For Gaussian distributions $P = \mathcal{N}(\mu, \sigma^2)$ and $Q = \mathcal{N}(\mu_{\text{old}}, \sigma_{\text{old}}^2)$, their probability density functions are:

$P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \\
Q(x) = \frac{1}{\sqrt{2\pi\sigma_{\text{old}}^2}} \exp\left(-\frac{(x - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2}\right) \\
\log P(x) = -\frac{1}{2} \log (2\pi) - \frac{1}{2} \log \sigma^2 - \frac{(x - \mu)^2}{2\sigma^2} \\
\log Q(x) = -\frac{1}{2} \log (2\pi) - \frac{1}{2} \log \sigma_{\text{old}}^2 - \frac{(x - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2}$

<br>

$$
\begin{split}
\text{KL}(P || Q) & = \int P(x) \left[ \log P(x) - \log Q(x) \right] dx \\
\text{KL}(P || Q) & = \int P(x) \left[ -\frac{1}{2} \log \sigma^2 - \frac{(x - \mu)^2}{2\sigma^2} + \frac{1}{2} \log \sigma_{\text{old}}^2 + \frac{(x - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2} \right] dx \\
\text{KL}(P || Q) & = \int P(x) \left[  \log \frac{\sigma_{old}}{\sigma} - \frac{(x - \mu)^2}{2\sigma^2} + \frac{(x - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2} \right] dx \\
\text{KL}(P || Q) & = \int P(x)  \log \frac{\sigma_{old}}{\sigma} - \int P(x) \frac{(x - \mu)^2}{2\sigma^2} + \int P(x) \frac{(x - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2} dx \\
\text{KL}(P || Q) & = \log \frac{\sigma_{old}}{\sigma} \int P(x) - \frac{1}{2\sigma^2} \int P(x) (x - \mu)^2 + \frac{1}{2\sigma_{\text{old}}^2} \int P(x) (x - \mu_{\text{old}})^2 dx \\
\text{KL}(P || Q) & = \log \frac{\sigma_{old}}{\sigma} - \frac{\sigma^2}{2\sigma^2} + \frac{\sigma^2 + 0 + (\mu - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2} \\
\text{KL}(P || Q) & = \log \frac{\sigma_{old}}{\sigma} - \frac{1}{2} + \frac{\sigma^2 + (\mu - \mu_{\text{old}})^2}{2\sigma_{\text{old}}^2} \\
\end{split}
$$

<br>

> $\int_{-\infty}^{\infty} P(x) = 1 \\
\int_{-\infty}^\infty P(x)(x - \mu)^2 \, dx = \text{Variance} = \sigma^2 \\
\int P(x)(x - \mu) = \text{Mean} = 0 \\
\scriptsize (x - \mu_{\text{old}})^2 = ((x - \mu) + (\mu - \mu_{\text{old}}))^2 = (x - \mu)^2 + 2(x - \mu)(\mu - \mu_{\text{old}}) + (\mu - \mu_{\text{old}})^2 \\
\int_{-\infty}^\infty P(x) (x - \mu_{\text{old}})^2 dx = \int P(x)(x - \mu)^2 + \int P(x)2(x - \mu)(\mu - \mu_{\text{old}}) + \int P(x)(\mu - \mu_{\text{old}})^2 \\
\int P(x) (x - \mu_{\text{old}})^2 dx = \sigma^2 + 0 + (\mu - \mu_{\text{old}})^2\int P(x) \\ $

---

<br>

1. **Asymmetry**: KL-divergence is not symmetric, $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$. This makes it a “divergence” rather than a true “distance.”
2. **Non-negativity**: $D_{\text{KL}}(P \| Q) \geq 0$, with equality if and only if $P = Q$ almost everywhere.
3. **Measures information loss**: It quantifies the information lost when $Q$ is used to approximate $P$.

<br>

---
- ### Example for discrete distribution

$$
P(x) = \left[ \frac{1}{3}, \frac{1}{3}, \frac{1}{3} \right]\\
Q(x) = \left[ \frac{1}{3}, \frac{1}{6}, \frac{1}{2} \right]
$$

The formula for KL-divergence is:

$$
D_{\text{KL}}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
$$

<br>

|$Outcome (x_i)$ | $P(x_i)$ | $Q(x_i)$ | $\frac{P(x_i)}{Q(x_i)}$ | $\log \frac{P(x_i)}{Q(x_i)}$ | $P(x_i) \log \frac{P(x_i)}{Q(x_i)}$|
|------------|------------|----|------------|------------|------------|
| $x_1$ | $\frac{1}{3}$ | $\frac{1}{3}$ | $\frac{\frac{1}{3}}{\frac{1}{3}} = 1$ | $\log 1 = 0$ | $\frac{1}{3} \cdot 0 = 0$|
| $x_2$ | $\frac{1}{3}$ | $\frac{1}{6}$ | $\frac{\frac{1}{3}}{\frac{1}{6}} = 2$ | $\log 2 = 0.693$ | $\frac{1}{3} \cdot 0.693 = 0.231$|
| $x_3$ | $\frac{1}{3}$ | $\frac{1}{2}$ | $\frac{\frac{1}{3}}{\frac{1}{2}} = \frac{2}{3}$ | $\log \frac{2}{3} = -0.405$ | $\frac{1}{3} \cdot -0.405 = -0.135$|

<br>

$$
D_{\text{KL}}(P \| Q) = 0 + 0.231 - 0.135 = 0.096
$$

---

<br>

As we mentioned that KL divergence is not symmetric $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$ .KL divergence is a kind of pseudomeasure of distance between two probability distributions.

We want to keep the new and old policies probability space close to each other.

$$
\large D_{KL}(\pi_\theta^{\text{updated}}(a|s)||\pi_\theta^{\text{current}}(a|s)\leq \delta
$$

<br>

> $\pi_\theta^{\text{updated}}(a|s)$ is the updated policy. we will use $\pi_\theta(a|s)$  <br>
$\pi_\theta^{\text{current}}(a|s)$ is the old/current policy. we will use  $\pi_\theta^{\text{old}}(a|s)$ <br>
$\delta$  is the maximum allowable KL-divergence.

<br>

- To incorporate this into the optimization, we move toward a substitute objective that uses the advantage function $A^{\pi_{\theta_{\text{old}}}}(s, a)$.

$$
\large D_{\text{KL}}(\pi_\theta \| \pi_\theta^{\text{old}}) = \mathbb{E}_{a \sim \pi_f} \cdot B
$$

$$
B = \left[ \log \frac{\pi_\theta(a|s)}{\pi_\theta^{\text{old}}(a|s)} \right]
$$
> $\pi_f$ is $\pi$ ff $\theta_o$, o stands for old


<br>

- we can use a Lagrangian formulation:

$$
\large J(\pi_\theta \| \pi_\theta^{\text{old}}) = \mathbb{E}_{a \sim \pi_f} \cdot B
$$

$$
B = \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right]
$$
> $\pi_f$ is $\pi$ ff $\theta_o$, o stands for old

<br>

> $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ is the ratio of new to old policy probabilities. <br>
$A^{\pi_{\theta_{\text{old}}}}(s, a)$ is the advantage function.

<br>

### **How we get there**

---
---
- Lagrangian formulation:

1. Objective Function: We aim to maximize the expected return based on the advantage function $A^{\pi_{\theta_{\text{old}}}}(s, a)$. The objective without constraints is:

$$
\mathbb{E}_{a \sim \pi_t} \left[A \right]
$$

$$
A = \left[ A^{\pi_{\theta_{\text{old}}}}(s, a) \right]
$$
> $\pi_t$ = $\pi_\theta$

2. Constraint: To ensure the new policy  $\pi_\theta$ does not deviate too far from the old policy $\pi_{\theta_{\text{old}}}$, we impose the KL-divergence constraint:

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\theta_{\text{old}}}) \leq \delta.
$$

- combine the objective and constraint into a Lagrangian

$$
\mathcal{L}(\theta, \lambda) = \mathbb{E}_{a \sim \pi_t} [A] - \lambda(D)
$$

$$
A = \left[ A^{\pi_{\theta_{\text{old}}}}(s, a) \right],\hspace{1cm} D =  \left( D_{\text{KL}}(\pi_\theta \| \pi_{\theta_{\text{old}}}) - \delta \right)
$$

> $\pi_t$: $\pi_\theta$ <br>
>  $\lambda \geq 0$ is the Lagrange multiplier enforcing the KL-divergence constraint. <br>
$\lambda D_{\text{KL}}(\pi_\theta \| \pi_{\theta_{\text{old}}})$ penalizes the objective if the KL-divergence exceeds  $\delta$.

<br>

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\theta_{\text{old}}}) = \mathbb{E}_{a \sim \pi_t} [L]
$$

$$
L = \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \right]
$$


To align the expectation with $\pi_{\theta_{\text{old}}}$ instead of  $\pi_\theta$, we rewrite the expectations using importance sampling:

$$
\mathbb{E}_{a \sim \pi_t} \left[ f(a) \right] = 
$$

$$
\mathbb{E}_{a \sim \pi^o_t} [L]
$$

$$
L = \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} f(a) \right]
$$

So, now we have this:
<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/lagrange.png">

---
---

<br>

$$
\large \pi_{\text{old}+1}= \underset{\pi}{\text{argmax}}J(\pi_\theta \| \pi_\theta^{\text{old}})
$$


### **Add the new term to derivarive**


$$
D_{\text{KL}}(\pi_\theta \| \pi_\theta^{\text{old}}) = \mathbb{E}{a \sim \pi_{\theta_{\text{old}}}} \left[ \log \frac{\pi_\theta(a|s)}{\pi_\theta^{\text{old}}(a|s)} \right]
$$

$$
J(\pi_\theta \| \pi_\theta^{\text{old}}) = \mathbb{E}{a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s, a) \right]
$$

> $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ is the ratio of new to old policy probabilities. <br>
$A^{\pi_{\theta_{\text{old}}}}(s, a)$ is the advantage function.

<br>

To handle the KL-divergence constraint, we penalize deviations beyond the limit $\delta$:

$$
\large L(\theta) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\theta_{\text{old}}}),
$$

> $\beta > 0$  is a penalty coefficient.

<br>

Taylor Expansion of KL-Divergence

<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/taylor_%20Expansion.png">

<br>

### Clipping objective


$$
\large J(\pi_\theta \| \pi_\theta^{\text{old}}) =
\begin{cases}
r(\theta) A^{\pi_{\theta_{\text{old}}}}(s, a), & \scriptsize \text{if } r(\theta) \in [1 - \epsilon, 1 + \epsilon], \\
\text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon) A^{\pi_{\theta_{\text{old}}}}(s, a), & \scriptsize \text{otherwise}.
\end{cases}
$$

<br>

>  $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ is the probability ratio. <br>
$\epsilon$ is a hyperparameter controlling the clipping range. <br>
$\text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon)$ ensures that the policy ratio does not deviate too far from the old policy.

<br>

- The objective using the PPO-clip variant is as follows:


$$
J(\pi_\theta \| \pi_\theta^{\text{old}}) = \text{min} \Biggl( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s, a) , F(\epsilon, A^{\pi_{\theta_{\text{old}}}}(s, a))\Biggr)
$$

<br>

$$
F(\epsilon, A) =
\begin{cases}
(1 + \epsilon) * A, & A \geq 0, \\
(1 - \epsilon)* A, & A < 0.
\end{cases}
$$

<br>

So:
- if $A \geq 0$:

$$
J(\pi_\theta \| \pi_\theta^{\text{old}}) = \text{min} \Biggl( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} , (1 + \epsilon)\Biggr) *  A^{\pi_{\theta_{\text{old}}}}(s, a)
$$

- if $A < 0$

$$
J(\pi_\theta \| \pi_\theta^{\text{old}}) = \text{min} \Biggl( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} , (1 - \epsilon)\Biggr) *  A^{\pi_{\theta_{\text{old}}}}(s, a)
$$

### Dynamic KL Penalty

<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/dynamic_KL_penalty.png">

$$
\begin{split}
L(\theta) & = e^{\left( \log \pi_\theta(a|s) - \log \pi_{\theta_{\text{old}}}(a|s) \right)} *  A^{\pi_{\theta_{\text{old}}}}(s, a) \\
\\
D_{\text{KL}}(\pi_\theta \| \pi_{\theta_{\text{old}}}) & = \Bigl( e^{\left( \log \pi_\theta(a|s) - \log \pi_{\theta_{\text{old}}}(a|s) \right)} - 1 \Bigr)^2
\end{split}
$$

<br>

---
$$
\log\left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \right) = \log \pi_\theta(a|s) - \log \pi_{\theta_{\text{old}}}(a|s)
$$

In order to restore the original scale of the probabilities, we must exponentiate the logarithmic difference($e^{\ln{f(x)}} = f(x)$):

$$
\exp\left( \log \pi_\theta(a|s) - \log \pi_{\theta_{\text{old}}}(a|s) \right) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
$$

---

## Model output

### Cartpole environment without bootstrapping
<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/cartpole.png">

### Cartpole environment wit bootstrapping
<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/cartpole_bootstrapping.png">

### LunarLander environment without bootstrapping
<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/lunarLander.png">

### LunarLander environment with bootstrapping
<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/lunarLander_bootstrapping.png">

<br>

> If you like to check references and find other RL algorithms implemetations check the [link](https://github.com/Mhddaraaa/start/tree/main/ReinforcementLearning)

