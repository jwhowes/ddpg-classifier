# DDPG Reinforcement Learning Classifier
Given a $H \times W$ image, the model iteratively observes a different $P_H \times P_W$ 
window of the image, using reinforcement learning to classify the image. The model 
must simultaneously balance classification accuracy against number of observations 
(preferring a smaller number).

The agent has two actions:
1. Termination
2. Continuation, parameterised by a window $x, y, w, h$

The agent is an autoregressive transformer which receives a sequence $(I_t, x_t, y_t, 
w_t, h_t)_{t=0}^{T - 1}$ of windows and predicts, for each item:
1. The cumulative termination reward $r_\mathcal{T}$
2. The cumulative continuation reward $r_\tilde{\mathcal{T}}$
3. The label distribution $\hat{y}_t$
4. The window logits $\mu_x, \mu_y, \mu_w, \mu_h$

The reward for a non-terminal state is $-\lambda$ where $\lambda > 0$ is a 
hyperparameter controlling the model's tolerance for uncertainty.

The critic is an autoregressive transformer which receives a sequence $(I_t, x_t, y_t, 
w_t, h_t, x_{t + 1}, y_{t + 1}, w_{t + 1}, h_{t + 1})_{t = 0}^{T - 2}$ and 
predicts the cumulative continuation reward for each item.

The critic is trained using MSE loss based on the actual cumulative reward.

The agent is trained based on three loss terms:
1. The cumulative and termination rewards are trained using the Bellman equation
2. The label distribution is trained using cross entropy loss
3. The window logits are trained to maximise the critic's prediction

## Randomised sampling
As hyperparameters, we receive an exploration probability schedule $\{\beta_t\}_{t=1}
^\infty$, and a logit probability schedule $\{\epsilon_t\}_{t=1}^\infty$.

At step $t$, after computing the agent's predictions, with probability $\beta_t$ we 
select from termination and continuation at random, else we select whichever has the 
highest cumulative reward prediction.

If continuation is chosen, we predict $x_t = \sigma(\mu_x + \epsilon_x)$ where 
$\sigma$ is the sigmoid function and $\epsilon_x \sim \epsilon_t$ is random noise.