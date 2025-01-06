---
title: "Data Engineering Fundamentals"
order: 1
book: "dMLSys"
layout: note
---

## Data Sources
---
- __input data__  : generally comes from user, can be dirty
- __system generated data__ : includes various types of logs and systems output.
	>[!caution] 
	> - logs are noizy -> hard to find signal (Logstash, Datadog, Log2.io)
	> - store large amount of data -> low-access / high-frequency access storage
- __intern data base__
- __thrid party data__



## Data Formats
---
==data serialization== : process of converting a data structure into a format that can be stored or transmitted and reconstructed later. [Wikipedia](https://en.wikipedia.org/wiki/Comparison_of_data-serialization_formats)
### Row-Major VS Column-Major Format
row-major -> [[Contiguous data]] makes it faster to read and write data observations (Numpy)
column-major -> better for analysis purposes ([[Pandas Python]])



## Data Models
---
### Relational Models
- __unordered data__ : order of the rows/columns is not important

relations should be normalized  (1FN, 2FN, etc...), it will help to reduce the redudancy & improve data integrity 
>[!caution]
>One major downside is that data can be massively spreaded and it can be costfull to run jointure operations

### NoSQL Models
- a document database : 
	- better locality, so easier to retrieve information
	- harder to join
- a graph database:
	- based on relations

### Structured VS Unstuctured Data
| Structured Data | Unstructured Data |
| --------------- | ----------------- |
| Data Warehouse  | Data Lake         |


## Data Storage Engines & Processing
--- 
Transactionnal VS Analytical Processing
- OLTP (Online Transactionnal Processing)
- OLAP (Online Analytical Processing)

## Modes of Dataflow
---
- Data passing through databases
- Data passing through services
	 - __Microservice architecture__ (==request-driven==)
	![[Pasted image 20250101170107.png|400]]
	 >[!caution]
	 > Request-driven data passing is synchronous so if one service is down, requests are blocked
- Data passing through real-time transport
	- broker system, in memory storage to broker data
	- ![[Pasted image 20250101170659.png|400]]
	- named event-driven
	- pubsub (Apache Kafka, Amazon Kinesis) & queue (Apache RocketMQ, RabbitMQ)

## Batch Processing VS Stream Processing
---
- batch features also known has static features (Spark & MapReduce)
- streaming feature = dynamic features (Apache Flink, KSQL, Spark Streaming)
