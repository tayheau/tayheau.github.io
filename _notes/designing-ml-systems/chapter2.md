---
layout: book
title: "Training Data"
book: "designing-ml-systems"
type: "chapter"
order: 2
---
# Training Data
---
## Sampling
### Non Probability Sampling
- convenience sampling : based on availability of the data
- snowball sampling : take one sample, then take the ones linked to it and so on
- judgment sampling : expert decision
- quota sampling: based on quota 
>[!caution]
>riddled with biaises

### Simple Random Sampling
can causes issues with some extreme occurences

### Stratified Sampling
divide population into subgroups (stratum) that are importants and then randomly pick from each subgroups.

### Weighted Sampling

### Reservoir Sampling
usefull to deal with streaming data : allows to keep k elements
- every element has an equal probability of being selected
- can stop the algorithm at any time and elements will be sampled with the correct probability
>[!example]
> 1 - put the firsts $k$ elements in the reservoir
> 2 - each incoming $n^{th}$ element generate $i$ so that $1 \leq i \leq n$
> 3 - if $1\leq i\leq k$ then replace the $k^{th}$ element with the new one

every element has $\frac{k}{n}$ chances to be selected.

>[!recursive proof]
> __init__

### Importance Sampling
Sample from one distribution when we only have access to another distribution
$$E_P[x] = \sum xP(x) = \sum Q(x)\frac{P(x)}{Q(x)}x = E_Q [\frac{P(x)}{Q(x)}x]$$

## Labeling
### Hands Labels
issues with label multiplicity 
==data lineage== : keep track of the origin of the data & labels

### Natural Labels
feed back loops length is an important criteria 

### Handling lack of labels 
#### Weak supervision 
Snorkel
based on heuristics :
- keyword heuristics
- regular expression
- database lookup
- output of other models

#### Semi Supervision
- self training
- similarity clustering
- perturbation based
most useful when the number of training labels is limited

#### Transfer Learning
- feature extraction
- fine tuning

#### Active Learning
label the samples that are the most helpful to your model according to some metrics or heuristics

## Class Imbalance
Accuracy is not the "holy" metric, it can be misleading in a context of data imbalance.
![](/_medias/Pastedimage20250101174751.png)
### ROC Curve
![](/_medias/Pastedimage20250101174837.png)
### Resampling
good for low dim data :
- undersampling (e.g : Tomek links)
- upsampling (e.g: SMOTE)

Many techniques are to computationally expensives for high-dimension data of high-dimensional feature space (Near&Miss, one-sided selection)

>[!caution]
>Never evaluate model on ressampled data, only train on it
### Sophisticated Sampling Techniques
- Two phases sampling:
	1 -  train on ressampled data
	2 - fine tune on original data
- Dynamic Sampling:
	oversample low-performing classes and oversample high-performing ones

### Algorithm level methods
- cost sensitive methods
- class balanced loss
- focal loss

## Data Augmentation
- Simple label-preserving transformation
- Perturbation (also simple label-preserving transformation) : used to trick model into making wrong prediction, also named "one pixel attack"
- Data synthesis
