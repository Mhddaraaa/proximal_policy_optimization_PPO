

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

<img align='center' width='1000' src="https://github.com/Mhddaraaa/proximal_policy_optimization_PPO/blob/main/lagrange.png">

<br>

- **Clipping objective**:

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

