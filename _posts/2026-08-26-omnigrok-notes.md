---
title: "omnigrok paper notes"
date: 2026-08-23
layout: post
---
# Omnigrok - Grokking Beyond Algorithmic Data
- There is an optimal weight norm for a model to reach generalisation
- There can be too high, too low or perfect weight norm for the model to be init with. In the two first case, grokking is the phenomenon of slowly reaching the optimal weight norm (Goldilocks zone)
- There is an influence of dataset size \\(N\\) on the Goldilocks width (and so on the generalisation time).
- Regularization inverse influence on generalisation time ($$ \approx \gamma^{-1}$$)
- There is a minimum dataset size to have to reach generalization
- Grokking can be deleted by containing weight norm constant.
- Grokking can happen on different nature of dataset / models.

## Polar coordinates setup

| trad setup                    | polar setup                                              |
| ----------------------------- | -------------------------------------------------------- |
| \\(\mathbf{w} \in \mathbb{R}^p\\) | \\((w, \mathbf{\hat{w}})\\) : weight norm, angular direction |
| \\(l(\mathbf{w})\\)  | \\(l(w, \mathbf{\hat{w}})\\)                                                         |

We define a reduced function 

$$
\tilde{f}(w) = f(\mathbf{w}^*(w)) \text{, where } \mathbf{w}^*(w) = \min_{||\mathbf{w}||_2=w}l_{train}(\mathbf{w})
$$


Through the paper they rely on $$\alpha = \frac{w}{w_0}$$, **multiplying factor of the default pytorch init**. $$w_0$$ and $$w$$ weight norm before and after the multiplication by $$\alpha$$. 

There is a spherical shell in the weight space (the "Goldilocks" zone), where generalization is better than outside this zone. 

![goldilocks](/assets/img/omnigrok/goldilocks.png){: #goldilocks style="width: 50%"}


## LU mechanism defined
**LU mechanism** is named after the observable shapes of test/train loss plotted against weight norm. It is identified as the cause of grokking.

The **U shape** represent the **test (generalization) loss** which is higher either both when $$w > w_c$$ or $$w < w_c$$ . This is often attributed to the bias-variance trade-off. The well known idea of the "generalization error" having a "U" shape was challenged bu the [[double descent]] phenomenon, but it can be recovered by changing the x-axis from the number of model params $$N$$ to the 2-norm of the model params 
$$
w=||\mathbf{w}||_2
$$ .

**L shape** represent training loss, with large error value at $$w<w_c$$ and small ones at $$w>w_c$$ because of overfitting solutions.

## Grokking dynamics
In **a large weight norm initialization** case, the model first quickly moves to the valley of overfitting solutions by minimizing the training loss. Implicit/explicit regularization will then drive the weight toward the **Goldilocks zone** ($$w \approx w_c$$). 
The "speed" of grokking depend of the regularization magnitude mostly $$t \approx \gamma^{-1}$$. 

A **small weight norm initialization** will always generalize fast, without consideration to regularization.

Grokking is not so common for a main reason: common init schemes typically have $$w < w_c$$ 

![](/assets/img/omnigrok/dynamics.png){: style="width: 50%"}

## Omnigrok: grokking for more interesting tasks
### mnist image classification 
> width 200 depth 3 ReLU MLP MSE loss AdamW optimizer lr 0.001 batch size 200
 
We reduce loss and error through reducing over angular direction of the weights 

$$
\tilde{l}_{train}(w, N) = l_{train}(\mathbf{w}^*, N), \space\space\space
\tilde{l}_{test}(w, N) = l_{test}(\mathbf{w}^*, N), \space\space\space
\mathbf{w}^* = argmin_{||\mathbf{w}||_2 = w}l_{train}(\mathbf{w}, N)
$$

![](/assets/img/omnigrok/datasize_landscape.png){: style="width: 50%"}

We observe 3 main points:
1. Larger init leads to grokking
2. Larger dataset size N leads to de-grokking (due to larger goldilocks zone)
3. Minimal dataset size to achieve generalization

To induce grokking in this mnist experimental setup, 2 decisions :
1. Reduce dataset size from N=60k to N=1k
2. Increase init scale ($$\alpha$$ > 1)

This leads to an other observation: generalization time increase as the dataset size decrease
