---
title: "Empirically challenging the norm as a sufficient descriptor of the weight init relevant to grokking"
date: 2026-08-31
layout: post
---

As we know, Liu et al. [[Omnigrok - Grokking Beyond Algorithmic Data]] empirically observe that the weight norm is related to grokking in a way that small weight norm at init 
will lead to faster grokking (generalisation) time.

But as mentionned in the comments of [this LessWrong post](https://www.lesswrong.com/posts/jusq6kyZ6XSrtW3Bf/paper-summary-omnigrok-grokking-beyond-algorithmic-data),
the norm is a strange metric to use.

Lets take the case of an unbiased ReLU network 

$$
f(W, X) = W_PReLU(...(W_2ReLU(W_1X)))
$$

and lets say we init $$W'$$, for $$\beta > 0$$

$$
W_P' = \frac{1}{\beta}W_P\qquad W_1'=\beta W_1
$$

then we have

$$
f_{W^{\beta}_0}(X) = f_{W_0}(X)
$$

but

$$
\|\|W_0^{(\beta)}\|\|_2 \ne \|\|W_0\|\|_2
$$

If the global norm was a sufficient causal or predictive descriptor of grokking time, then this 'function-preserving' rescaling that change global norm 
should also induce a (monotonic) change in grokking time.

Taking a [mnist grokking setup](https://colab.research.google.com/drive/1wLkyHadyWiZSwaR0skJ7NypiYKCiM7CR?usp=sharing) from the paper, 
using $$\beta \in \left[0.25, 0.5, 1.0, 2.0, 4.0\right]$$ and averaging each beta values accross 10 differents rng seeds, we have the following results

![](/assets/img/challenging/mnist_function_preserving_rescale.png){: style="width: 50%"}

We we can clearly observe that grokking happens in the same time window for every $$\beta$$, so there is no obvious relation 
between grokking time and weight norm.

I think we can affirm that **global weight norm is not a sufficient descriptor of the initialization of the wieghts relevant to grokking**.

Which support kumar's [ [1] ](#1)  take that global weight norm might just be an incomplete proxy.

Im wondering if the initial NTK can be a sufficient observation de determin if we are going to grok, 
we should do the same experiment and do a NTK similarity matrix using the CKA and also see if beta change the task alignment $$CKA(K_0^\beta,YY^\top)$$ (hopefully not).
 
 Some results with the same $$\beta$$ setup, plotting initial NTK metrics:

![](/assets/img/challenging/mnist_beta_initial_ntk_metrics.png){: style="width: 50%"}

![](/assets/img/challenging/mnist_beta_pairwise_ntk_cka.png){: style="width: 30%"}

We can observe that the matrix and the target alignment have very small variance scale.
Given that all networks grok approx at the same time, it doesnt dismiss NTK based metrics as efficients one to look at to init weights.

## References
<a id="1">[1]</a> Kumar et al. *Grokking as the Transition from Lazy to Rich Training Dynamics.* 2023 [cs.LG](https://arxiv.org/abs/2210.01117)
