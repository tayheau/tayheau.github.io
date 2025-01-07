---
layout: book
title: "Feature Engineering"
book: "designing-ml-systems"
type: "chapter"
order: 3
---
# Feature Engineering
---
## Common feature engineering operations
- __Handling Missing Values__:
	- (MNAR) Missing Not At Random : missing because of the value itself
	- (MAR) Missing At Random : missing not due to the value itself but because of an other observable one
	- (MCAR) Missing Completely At Random : no pattern in when the value is missing

- __Feature Scaling__
- __Discretization__
- __ Encoding Categorical Features__
	- Hashing Trick
- __Feature Crossing__
- __Discrete and continuous positional embeddings__

## Data Leakage
### reasons : 
- splitting time-correlated data randomly and not by time
- scalling before splitting
- filling in missing data with stats from the test split
- poor handling of data duplication before splitting 
- group leakage
- leakage from data generation process

### detecting data leakage
- correlation with label (alone or grouped)
- ablation studies

## Feature Importance
example for traditional ML : XGBoost
for model-agnostic methods, look into SHAP (InterpretML)
