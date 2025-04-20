# DDPG Reinforcement Learning Classifier
Given a $H \times W$ image, the model iteratively observes a different $P_H \times P_W$ 
window of the image, using reinforcement learning to classify the image. The model 
must simultaneously balance classification accuracy against number of observations 
(preferring a smaller number).

A state is a sequence of tuples $s_t = ((I_t, x_t, y_t, w_t, h_t))_{t=0}^t$ where 
$I_t$ is the crop based on top left position $x_t, y_t$, width $w_t$ and height $h_t$, 
resized to the model's patch size $P_H \times P_W$.

Note that $x_0 = y_0 = 0$, $w_0 = 1$, and $h_0 = 1$. (i.e. the first crop is always the 
full image).

A classifier $f_\theta(\hat{y} | s_t)$ predicts a label distribution $\hat{y}$ given 
the state $s_t$.

An actor $\pi_\gamma(\alpha_{t+1}, \mathcal{N}_{x, t+1}, \mathcal{N}_{y, t+1}, 
\mathcal{N}_{w, t+1}, \mathcal{N}_{h, t+1} | s_t)$ predicts a termination 
probability $\alpha_{t+1}$ and four Gaussian distributions. At state $s_t$, the model 
has the (continuous) set of actions 
$\{(x, y, w, h) | x, w, y, h \in [0, 1]^4\} \cup \{\mathcal{T}\}$ where $\mathcal{T}$ 
corresponds to termination.

The reward function is defined like so:
$$
\begin{equation}
\mathcal{R}(s_t, a_{t+1}) = \begin{cases}
-\log \frac{\exp(\hat{y}_y)}{\sum_{y'} \exp(\hat{y}_{y'})}, & a_{t+1} = \mathcal{T}\\
-\lambda, & \textrm{otherwise}
\end{cases}
\end{equation}
$$
where $\lambda > 0$ is a hyperparameter meant to encourage early termination. 
Typically, $\lambda = \frac{\log 2}{k}$ where $k \in \mathbb{N}$ controls the desired 
length. The idea behind this is that once $t \geq k$ it becomes beneficial to 
terminate once the classifier's confidence exceeds $0.5$ (assuming its classification is 
correct).

A critic $Q_\varphi(s_t, a_{t+1})$ predicts the total reward $\sum_{t' = t + 1}^T 
\mathcal{R}(s_{t'}, a_{t'})$.

## Stochastic Action Policy
We use an exploration probability schedule $\beta_t$, and a noise schedule $\epsilon_t$,
where each $0 \leq \beta_t \leq 1$ and each $\epsilon_t$ is a random noise generation 
function.

With probability $\beta_t$ we select our action from $\{\mathcal{T}, (x, y, w, h)\}$ 
where $x \sim \mathcal{N}_{x, t + 1}$ etc. with uniform probability.

Otherwise, if $\alpha_{t+1} > 0.5$, we select $\mathcal{T}$ else we select $(x, y, w, h)$.

## Training
