---
title: "PCA, but probabilistic"
date: 2025-04-10
layout: post
---

Multiples hidden Markov models can be seen as variants of one underlying generative model : discrete time linear dynamical systems with Gaussian Noise[^1] .
A so named system is defined by :

$$ 
\begin{align}
\mathbf{x}_{t+1} &= \mathbf{A}\mathbf{x}_t + \mathbf{w}_t = \mathbf{A}\mathbf{x}_t + \mathbf{w}_\bullet & \mathbf{w}_\bullet \sim \mathcal{N}(0, \mathbf{Q}) \tag{1a} \\

\mathbf{y}_t &= \mathbf{C}\mathbf{x}_t + \mathbf{v}_t = \mathbf{C}\mathbf{x}_t + \mathbf{v}_\bullet & \mathbf{v}_\bullet \sim \mathcal{N}(0, \mathbf{R}) \tag{1b}
\end{align}
$$
where, we resume the main goal *first-order Gaussian-Markov random process* $\mathbf{x}_t$ to be an informative lower dimensional projection of the *observation sequence* $\mathbf{y}_t$.
We also define the gaussian noise to be 0 mean without loss of generality [^1]$

In the case of system identification (learning), we can perform the EM algorithm to obtain the values of all the parameters $\{\mathbf{A}, \mathbf{C}, \mathbf{Q}, \mathbf{R}, \mathbf{\mu}_1, \mathbf{Q}_1 \}$ given the observed data $\{\mathbf{y}_1 ... \mathbf{y}_\tau\}$ that maximises the likelihood of $P(\{\mathbf{y}_1...\mathbf{y}_\tau\})$.



Doing so is equivalent to maximizing $F(Q, \theta) \le \mathcal{L}(\theta)$ defined by:
$$
\begin{align}
F(Q, \theta) &= \int_\mathbf{X}Q(\mathbf{X})\log P(\mathbf{X,Y|\theta})d\mathbf{X} - H(Q) \tag{2a}\\
&=\mathcal{L}(\theta) - D_{KL}(q||p(\mathbf{X}| \mathbf{Y}, \theta)) \tag{2b}
\end{align}
$$
with $Q$ being a random distribution for the hidden variable $\mathbf{X}$. It can also be seen as the difference between the *$q(x)$ and $p(x|y)$ Kullback-Liebler divergence and $\mathcal{L}(\theta)$*.

We can then express the EM-algorithm in terms of the $F$ function [^2]:
$$
\begin{array}{ll}
\text{E Step:} & \text{Set } Q^{(t)} \text{ to the } Q \text{ that maximizes } F(Q, \theta^{(t-1)}). \\
\text{M Step:} & \text{Set } \theta^{(t)} \text{ to the } \theta \text{ that maximizes } F(Q^{(t)}, \theta).
\end{array} 
$$
based on the two following lemmas
>[!tldr] Lemma 1
>__*For a fixed value $\theta$, there is a unique distribution $Q_\theta$ that maximises $F(Q, \theta)$ given by $Q_\theta(x) = P(x|y, \theta)$*__
>>[!note] Proof
>>$$
>>\begin{align}
>>&& \frac{\partial}{\partial Q}F(Q, \theta) &= 0\\
>>\Leftrightarrow && \frac{\partial}{\partial Q} \bigl[\int q(x)\log p(x, y|\theta) dx + \int q(x)\log\frac{1}{q(x)}dx +\lambda\{\int q(z)dz - 1\} \bigr] &= 0\\
>>\Leftrightarrow && \log p(x, y|\theta) - \log q(x) - \lambda - 1 &=0\\
>>\Leftrightarrow && \log q(x) &= \log p(x, y|\theta)  - \lambda - 1\\
>>\Leftrightarrow && q(x) &\propto p(x, y| \theta)\\
>>\Leftrightarrow &&\text{normalizing, } q(x) = p(x|y, \theta)
>>\end{align}
>>$$

>[!tldr] Lemma 2
>__**If $Q(x) = P(x|y, \theta)$ then $F(Q, \theta) = \mathcal{L}(\theta)$**__.
>>[!note] Proof
>>If $Q(x) = P(x|y, \theta)$
>>$$
>>\begin{align}
>>F(Q, \theta) &= \int q(x)\log p(x, y|\theta)dx - \int q(x)\log q(x) dx\\
>>&=E_Q\bigl[\log p(x, y|\theta)\bigr] - E_Q\bigl[\log q(x)\bigr]\\
>>&= E_Q\bigl[\log p(x, y|\theta)\bigr] - E_Q\bigl[\log p(x|y, \theta)\bigr]\\
>>&= E_Q\bigl[\log\frac{p(x, y|\theta)} {p(x|y, \theta)}\bigr]\\
>>&=\log p(y|\theta)
>>\end{align}
>>$$

Then at each end of a **E Step**, we have $F = \mathcal{L}$, maximizing $F$ at the **M Step** is equivalent to maximizing $\mathcal{L}$.

Combining thoses two lemmas, we come up with the fact that 
>[!tldr] Theorem
>**If $F(Q, \theta)$ as a local (global) maximum at $Q^*$ and $\theta^*$ then $\mathcal{L(\theta)}$ as a local (global) maximum at $\theta ^*$ too.**
>>[!note] Proof
>>For $Q = Q_\theta$ (in our case at the end of the **E Step**), we have $\mathcal{L}(\theta) = F(Q_\theta, \theta) = \log p(y|\theta)$.
>>Then at the end of the **M Step**, we have $\theta ^*$ so that $F(Q_{\theta ^*}, \theta ^*)$ is local maximum.
>>We then suppose that $\exists \space \theta^\tau$ near $\theta ^*$ so that $\mathcal{L}(\theta ^\tau) > \mathcal{L}(\theta ^*).$ Meaning so will implies that $F(Q_{\theta ^\tau}, \theta ^\tau) > F(Q_{\theta ^*}, \theta ^*)$ but since $Q_\theta$ varies continuously with $\theta$, it will imply that $Q_{\theta ^\tau}$ is near $Q_{\theta ^*}$, being in contradiction that $Q_{\theta ^*}$ and $\theta ^*$ are *local* maximums. The same logic goes for the global.

We can then justify an incremental version of the **EM Algorithm** based on sufficient statistics vector $s(x, y) = \sum_i s_i (x_i, y_i)$[^2]:
$$
\begin{array}{ll}
\text{E Step: } &\text{Chose some data item } i \text{ to be updated.}\\
&\text{Set } \tilde{s}_j^{(t)} = \tilde{s}_j^{(t-1)} \space \forall i \ne j\\
&\text{Set } \tilde{s}_i^{(t)} = E_{Q_i}\bigl[s_i(x_i, y_i)\bigr] \text{ with } Q_i = p(x_i | y_i, \theta ^{(t-1)})\\
&\text{Set } \tilde{s}^{(t)} = \tilde{s}^{(t-1)} - \tilde{s}_i^{(t-1)} + \tilde{s}_i^{(t)}\\
\text{M Step: } &\text{Set }\theta ^t \text{ to the maximum likelihood given } \tilde{s}^{(t)} 
\end{array}
$$

In the case of PCA, we are in a case with no temporal dependence, indeed the hidden variable $x$ has no dynamics and so the matrix $A$ is the zero matrix.

In doing so, we end up with the following system:

$$
\begin{align}
\mathbf{x}_\bullet &= \mathbf{w}_\bullet & \mathbf{w}_\bullet \sim \mathcal{N}(0, Q) \tag{2a}\\
\mathbf{y}_\bullet &= C\mathbf{x_\bullet} + \mathbf{v}_\bullet & \mathbf{v}_\bullet \sim \mathcal(0, R) \tag{2b}
\end{align}
$$
There is a persisting degeneracy in the system : all the information in $Q$ can be placed into $C$ (and $A$ in the original system) : meaning that different configurations can lead to same results. To solves this, we are going to set $Q$ as the identity matrix without loss of generality : 
$$
\begin{align}
&\text{We have } \mathbf{w}_\bullet \sim \mathcal{N}(0, Q) \text{ and we want } \textbf{w}_\bullet ^{'} \sim \mathcal{N}(0, I).\\
&\text{We are then going to apply a whitening transformation to } \textbf{w}_\bullet  \text{ by } \textbf{w}_\bullet ^{'} = T\textbf{w}_\bullet \text{ so that } \mathrm{Var}(\mathbf{w}_\bullet ^{'}) = I.\\
\end{align}
$$
$$
\begin{align}
\mathrm{Var}(\mathbf{w}_\bullet ^{'}) &= I\\
\Leftrightarrow TQT^T &=I\\
\Leftrightarrow TE\Lambda E^TT^T &= I \text{ since } Q \text{ semi-positive definite and diagonal}\\
\Leftrightarrow T &= \Lambda ^{-1/2}E^T \text{ since } E \text{ orthogonal } 
\end{align}
$$
$$
\begin{align}
&\text{Then we have } \mathbf{x}^{'}_\bullet = \mathbf{w}^{'}_\bullet = T\mathbf{w}_\bullet\\
&\text{So } \mathbf{y}^{'}_\bullet = C\mathbf{x}^{'}_\bullet +  \mathbf{v}_\bullet = C^{'}\mathbf{x}_\bullet +  \mathbf{v}_\bullet \text{ with } C^{'}=C\Lambda ^{-1/2}E^T
\end{align} 
$$
We can view then, in the case of PCA, the $C$ matrix as the principal subspace transformation matrix.

We then end up with, by completing the squares on (2a) and (2b): 
$$
\begin{align}
p(\mathbf{x}_\bullet) &\sim \mathcal{N}(0, I) \\
p(\mathbf{y}_\bullet|\mathbf{x}_\bullet) &\sim \mathcal{N}(C\mathbf{x}_\bullet, R)\\
p(\mathbf{y}_\bullet) &\sim \mathcal{N}(0, CC^T + R)\\
p(\mathbf{x}_\bullet|\mathbf{y}_\bullet) &\sim \mathcal{N}(\beta \mathbf{y}_\bullet, I - \beta C ), \space \beta = C^T(CC^T+R)^{-1}
\end{align}
$$

But, even if it's a loss of generality, we have to limit R in some way, either way, $R$ could be the sample covariance while $C = 0$. We then put $R = \lim_{\epsilon \rightarrow 0} \epsilon I$.
So we end up with :
$$
p(\mathbf{x}_\bullet|\mathbf{y}_\bullet) \sim \mathcal{N}((C^TC)^{-1}C^T\mathbf{y}_\bullet, 0)
$$

We have then the following **EM Algorithm**[^3]:
$$
\begin{array}{ll}
\text{E Step: }&\mathbf{x} = (C^TC)^{-1}C^T\mathbf{y}\\
\text{M Step: }&C^{(new)} = YX^T(XX^T)^{-1}
\end{array}
$$
Since 
$$
\begin{align}
\frac{\partial}{\partial C}\log p(\mathbf{X}, \mathbf{Y}|\theta ) &= 0\\
\Leftrightarrow\frac{\partial}{\partial C} \bigl[-\frac{1}{2}\sum ||\mathbf{y}_n-C\mathbf{x}_n||_2 \bigr] &=0
\end{align}
$$
or 
$$
\begin{align}
||\mathbf{y}_n-C\mathbf{x}_n||_2 =\mathbf{y}_n^T\mathbf{y}_n - 2\mathbf{y}_n^TC\mathbf{x}_n+ \mathbf{x}_n^T C
^TC\mathbf{x}_n = J(C) \tag{3}
\end{align}
$$
Using Taylor-Series:
$$
\begin{align}
\frac{d}{d C}J(C) &= \mathrm{tr}\Bigl(\frac{\partial J(C)}{\partial C}^T dC\Bigr) + \epsilon(1)\\
&=2\mathbf{y}_n^TdC\mathbf{x}_n+\mathbf{x}_n^TdC^TC\mathbf{x}_n + \mathbf{x}_n^TC^TdC\mathbf{x}_n\\
&=-2\mathrm{tr}\Bigl(\mathbf{X}\mathbf{Y}^TdC\Bigr) + 2\mathrm{tr}\Bigl(\mathbf{X}\mathbf{X}^TC^TdC\Bigr)\\
\end{align}
$$
and so 


$$
\begin{align}
\frac{\partial}{\partial C}J(C) &= 0\\
\Leftrightarrow \mathbf{X}\mathbf{Y}^T &= \mathbf{X}\mathbf{X}^T
C^T\\
\Leftrightarrow C &= YX^T(\mathbf{X}\mathbf{X}^T)^{-1}
\end{align}
$$

The two sufficient statistics are then :
$$
\begin{align}
\mathbf{Y}\mathbf{X}^T\tag{4}\\
\mathbf{X}\mathbf{X}^T\tag{5}
\end{align}
$$
Then here are the incremental sufficient statistic necessary for the EM Algorithm is:
$$
\begin{array}{ll}
\text{E Step: } & \tilde{s_1}^{(t)} = E_{P(\mathbf{x}|\mathbf{y})}[\mathbf{y_i\otimes_outer x_i^T}] = \mathbf{y}_i\otimes_{\mathbf{outer}} \Bigl\{(C^TC)^{-1}C^T\mathbf{y}_i\Bigr\}^T = \mathbf{y}_i\otimes_{\mathbf{outer}}\hat{x}_i^T\\
&\tilde{s_2}^{(t)} = \hat{x}_i^T\otimes_\mathbf{outer}\hat{x}_i^T\\
\text{M Step: } &C^{new} = \tilde{s_1}^{(t)}\tilde{s_2}^{(t)}
\end{array}
$$

[^1]: Roweis, S. T., & Ghahramani, Z. (1999). A unifying review of linear Gaussian models. _Neural Computation_, _11_(2), 305–345. https://doi.org/10.1162/089976699300016674

[^2]: Neal, Radford & Hinton, Geoffrey. (2000). A View Of The Em Algorithm That Justifies Incremental, Sparse, And Other Variants. Learning in graphical models. 89. 10.1007/978-94-011-5014-9_12. 

[^3]: Sam Roweis. 1997. EM algorithms for PCA and SPCA. In Proceedings of the 11th International Conference on Neural Information Processing Systems (NIPS'97). MIT Press, Cambridge, MA, USA, 626–632.

