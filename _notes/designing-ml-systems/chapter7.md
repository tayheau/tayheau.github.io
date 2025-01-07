---
layout: book
title: "Continual Learning & Test in Production"
book: "designing-ml-systems"
type: "chapter"
order: 7
---
# Continual Learning & Test in Production
---
## Continual Learning
champion VS challenger model
- __stateless retraining__: retraining the model from scratch each time
- __stateful retraining__: conitnue retraining on new data

continual learning is about setting up an infrastructure in a way that allow the update of models whenever needed and deploy it quickly.

- __model iteration__: new features or different model architecture
- __data iteration__ : new data, same model

Continual Learning is good against data shift. It also help overcome the ==continuous cold start problem==
### Continual Learning challenges
- __fresh data access challenge__ (but with streaming natural labelled data, it's label computation)
- __evaluation challenge__ 
	- biggest challenge. The more updates, the more likely it will faill
	- more susceptible to coordinated manipulation & adversarial attack  eg [[Tay - le Chatbot Raciste]]
	- evaluation takes time
- __algorithm challenge__
	- only affects matrix-based & tree-based models that are updated very fasts
	- much easier to adapt NN to continual learning paradigm than matrix or tree based algorithms.
### How often to update your models
- value of data freshness : test on different slides of window date
- models iterations vs data iteration case
## Test in Production
- need a mixture of online & offline predictions
- offline not enough ? 
	- 2 majors test types for offline eval : test splits & backtest
		- static test split to benchmark and compare different model performances
		- backtest : testing model on data from specific period of time in the past. Not suffisent, can be an issue with the new data. Always use a static test set in addition as a sanity check.
- Data shitfs
	- models does well on data from 1h but what about later ? You have to deploy to check
### Shadow Deployment
1. deploy candidate model in parallel with the existing one
2. for each incoming requests, route it to both models to make predictions, but only serve the existing model's predicitons
3. log the candidate predictions for analysis purpose
### A/B Testing
1. deploy candidate model in parallel with the existing one
2. alternatively route requests to one or another (has to be random so there is no selection biais, should run enough samples to be confident about the outcome)
3. analyse the logs (can do two-sample test to determine which model is better)
### Canary Release
candidate model (canary) is deployed to a subgroup of users
### Interleaving Experiments
expose users to both models. Needs less population than A/B testing 
	- the two models recommandations should be as likely so there is no biais in the exposition -> that's why we use ==team-draft interleaving==
### Bandits
comes from gambling, more data efficient than A/B testing
eploration & exploitation
### Contextual Bandits
also called "one shot reinforcment"
bandits -> determine the payout (prediction accuracy) of each models
contextural bandits -> determine the payout of each actions

All the process of model evaluation should be clearly determined by the team : which test, in which order on which data. Better, it should be automatized and kicked off when there is a new update. Should be similar to continuous integration developpment (CI/CD)
