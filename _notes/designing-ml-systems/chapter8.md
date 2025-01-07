---
layout: book
title: "Infrastructure & Tooling for MLOps"
book: "designing-ml-systems"
type: "chapter"
order: 8
---
# Infrastructure & Tooling for MLOps
---
![](/_medias/Pastedimage20250106010338.png)
## Storage & Compute Layer
- all the compute ressources a company has access to and the mechanisms to determine how theses ressources can be used
- can be used into smaller compute units to be used concurrently
__threads__ : to execute a job
__instance__ : "permanent" unit
## Dev Environment
IDE, versionning, CI/CD, SSH, containers
## Ressource Managment
- __CRON__ : run script at a predetermined time & tells if failed or succeded
- __SCHEDULER__ : cron that handle dependencies (needs to know the available ressources)
- __ORCHESTRATORS__
## DS Workflow managment
specify workflows as DAGs where each step is an edge
Airflow, Argo, Perfect, Kubeflow, Metaflow
## ML Plateform
shared set of tools for ML Deployment
- __Model Store__ : 
	- model definition
	- model parameters
	- featurize & predict functions
	- dependencies
	- data (version or endpoint)
	- model generation point
	- experiment artifacts
	- tags
ML Flow is the most popular

- __Feature Store__ 
	- feature managment
	- feature consistency
	- feature computation
