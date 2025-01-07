---
layout: book
title: "Data Distribution Shifts & Monitoring"
book: "designing-ml-systems"
type: "chapter"
order: 6
---
## Causes of the ML System failures
operational expectation violation & ML performance expectation violation 
## Software system failures
- dependency failure
- deployment failure
- hardware failure
- downtime / crashing
## ML-Specific failures
- production data + training data are not of the same distribution 
- edge cases
- degenerate feedback loops : can be the exposure biais, result from user feedback. Can cause a system to be more homogeneous over time.
	- use randomization to correct
	- use positional feature
## Data Distribution Shifts
data distribution changing over time
train : source distribution
inference : target distribution

according to bayes : $P(X \cap Y)$ = $P(Y|X)P(X)$
- __covariance shift__  : $P(X)$ changes but not $P(Y|X)$ (input dist changes but not pred's)
- __label shift__ : $P(Y)$ changes but not $P(X|Y)$ 
- __concept drift__ : $P(Y|X)$ changes but not $P(X)$

__other general data distribution drifts:__
- feature change
- label schema change


data shift is an issue only if it causes the model's performance to degrade, you have to :
- monitor accuracy related metrics
- monitor $P(X)$, $P(Y)$ and conditional distributions $P(X|Y)$ and $P(Y|X)$
- use statistics such as min, max, var, mean, etc... of the distribution but they are not enough -> two sample-test

shifts ca, happen across two dimensions : spatial or temporal -> to detect temporal shifts, a common approach is to threat input data as time-series data, cumulative vs sliding statistics
## Monitoring & Observating
__monitoring__ : act of tracking, measuring and logging different metrics
__observability__: setting up the system to be monitored 
### operational metrics (SYSTEM):
- network
- machine
- application
(SLOs) or Service Level Agrements (SLAs), e.g is up if median latency < 200ms & 99th percentil < 2s
### ML Specific Metrics
- row inputs 
- features : check expectations to detect shifts in distribution 
- prediction : can detect shifts (since low-dim -> two sample test)
- accuracy : can be feedback

Great Expectation & Deequ by AWS for feature monitoring

Can also check data shift with two-sample test

### Monitoring Toolbox
- metrics logs traces from monitoring dev POV
- logs dashboard alerts for monitoring users
- for microservice architecture : ==distributed tracing==, where each process is given an unique ID in the logs + all metadata.

KSQL or FlinkSQL for streaming data
