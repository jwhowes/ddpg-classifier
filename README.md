# DDPG Reinforcement Learning Classifier
Given a $H \times W$ image, the model iteratively observes a different $P_H \times P_W$ 
window of the image, using reinforcement learning to classify the image. The model 
must simultaneously balance classification accuracy against number of observations 
(preferring a smaller number).

## Architecture and Training
The agent $\pi_\theta$ is an autoregressive transformer which recieves a sequence $(I_t, a_t)_{t=1}^{T - 1}$. For each $I_t, a_t$, $\pi_\theta$ 
predicts the termination/continuation rewards $\mathcal{R}_\mathcal{T}, \mathcal{R}_\tilde{\mathcal {T}}$, 
the logits for the next window $\mu_x, \mu_y, \mu_w, \mu_h$, and the label 
distribution $\hat{y}$.

The critic network $Q_\varphi$ which receives a sequence $(I_t, a_t, a_{t + 1})_{t=1}^
{T - 1}$. For each $I_t, a_t, a_{t + 1}$, $Q_\varphi$ predicts the cumulative reward 
$\sum_{i=1}^{T - t - 1} \gamma^{i - 1} \mathcal{R}(s_{t + i})$.

The reward function is defined as:
$$
\begin{equation}
\mathcal{R}(s) = \begin{cases}
\log \frac{\exp(\hat{y}_y)}{\sum_{y'} \exp(\hat{y}_{y'})}, & s \textrm{ is terminal} \\
-\lambda, & \textrm{otherwise}
\end{cases}
\end{equation}
$$
where $\lambda > 0$ is a hyperparameter controlling the agent's tolerance for uncertainty.