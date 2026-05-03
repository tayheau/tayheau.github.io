---
title: "Neural Tangent Kernel and Lazy regime."
date: 2026-05-03
layout: post
---
* TOC
{:toc}  

The Lazy Regime is a phenomenon that can be observed during the training the overparametrized DNN: the network remains near linearized form undergoing minimal changes in the parameter space, while the loss still converge toward 0.

To explore this phenomenon, let's study the evolution of the parameters $$\theta$$ through time by taking 

$$ \begin{align}
&\theta_0 \text{ at init} \\
&\theta_T \text{ at } loss=0
\end{align} 
$$

Then, our 'accurate' prediction $$\hat{y}$$ is given by 

$$ \hat{y} = f(\theta_T) $$

Since we work in the lazy regime context, $$\theta$$ is theoricaly supposed to stay still, we can then do a first order taylor expansion on $$\hat{y}$$:

$$\begin{align}
\hat{y} = f(\theta_T) &= f(\theta_0) + (\theta_T - \theta_0)\nabla_{\theta}f(\theta_0) \\
&=f(\theta_0)+\Delta_{\theta}\nabla_{\theta}f(\theta_0)
\end{align} 
$$


We can see that it becomes a linear function in $$\theta$$ that can be expressed as 

$$ \hat{y} = \Phi^T\theta + c $$

 with the feature space being the gradient vector at init $$\phi(x)=f(x, \theta_0)$$.
