---
layout: book
title: "Model Deployment"
book: "designing-ml-systems"
type: "chapter"
order: 5
---
# Model Deployment
---
## Batch prediction VS Online prediction 
### ML Deployment Myths
- you only deploy one or two models at a time
- if we don't do anything, model performance remain the same
- you won't need to update your model as much
- most ML Engineers don't need to worry about scale

### Batch prediction
![](/_medias/Pastedimage20250104193827.png)
### Online prediction using batch features
![](/_medias/Pastedimage20250104194201.png)
### Online Prediction (Streaming)
![](/_medias/Pastedimage20250104194821.png)

## Model Compression
- low rank factorization, e.g : compact conv filters
- knowledge distillation : student & teacher model
- pruning : by zeroing params, but can introduce biais
- quantization : using fewer bits to represent parameters
- [Roblox Bert Case Study](https://medium.com/@quocnle/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26)

## ML on the Cloud VS On Device
### ML Optimization on Edge 
- vectorization
- parallelization
- loop tiling
- operator fusion
- graph optimization
