# Adaptive Multilevel Community Detection in Dynamic Networks

## Modularity Optimization, Temporal Stability and Computational Scalability

**MSc Research Project | IIT Bombay**

**Student:** Dipendu Pal (25N0465)

**Duration:** August–November 2026

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-blue)

## Abstract

This project studies community detection in static and dynamic
networks using modularity optimization.

The main objective is to reduce computational work by using
previous community information when new nodes and edges arrive,
while maintaining community quality and temporal stability.

Static Louvain, warm-start Louvain and adaptive multilevel
Louvain are compared using synthetic and real-world networks.

## Dissertation and Presentation

- [Final Dissertation](_25N0465.pdf)
- [Final Presentation](25N0465_.pptx)

## Mathematical Foundations

- Graph theory
- Girvan-Newman algorithm
- Modularity and null models
- Spectral community detection
- NP-hardness of modularity maximization
- Louvain algorithm
- Leiden algorithm
- Stochastic Block Model
- Adjusted Rand Index

## Mathematical Contributions

1. Local modularity gain
2. Modularity preservation under aggregation
3. Deterministic affected-region bounds
4. Probabilistic affected-region analysis
5. Connectivity refinement theorem
6. Quality-efficiency-stability optimization

The mathematical analysis combines established results
with the algorithmic and experimental work of this project.

## Datasets

| Dataset | Nodes | Undirected edges |
|---|---:|---:|
| Zachary Karate Club | 34 | 78 |
| Email-Eu-core | 1005 | 16064 |
| French High School | 327 | 5818 |
| Temporal Email-Eu-core | 986 | 16064 |

## Experiments

- Static SBM community recovery
- Dense SBM probability grid
- Empirical community recovery boundary
- Community-focused node arrivals
- Bridge-node arrivals
- Controlled real-world arrivals
- Degree-matched experiments
- Temporal community detection
- Connectivity refinement
- Computational scalability
- Hybrid Louvain with periodic resets
- Pareto analysis

## Main Findings

- 80,000 SBM realizations across 800 probability combinations.
- Maximum mean ARI of approximately 0.9987.
- Empirical inverse-type recovery boundary.
- Approximately 88% reduction in node visits on Email-Eu-core.
- Approximately 91% reduction in candidate evaluations.
- Approximately 25% reduction in mean runtime.
- Successful refinement of all 38 disconnected partition records.
- Scalability experiments up to 9,600 base nodes.
- Periodic resets reduced accumulated differences from static Louvain.

The empirical recovery boundary and scaling exponents
are specific to the experimental settings.

## Repository Structure

    Community-Detection-in-Networks/
    |
    |-- 01_datasets/
    |-- 02_previous_work/
    |-- 03_algorithms/
    |-- 04_experiments/
    |-- 05_results/
    |   |-- raw/
    |   |-- summary/
    |-- 06_figures/
    |-- 07_report/
    |-- 08_validation/
    |-- _25N0465.pdf
    |-- 25N0465_.pptx
    |-- README.md

## Reproducibility

The repository contains the algorithms, experimental scripts,
raw results, summary tables, figures and validation outputs.



## Data Sources

- [Zachary Karate Club](https://www-personal.umich.edu/~mejn/netdata/)
- [SNAP Email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html)
- [SNAP Temporal Email](https://snap.stanford.edu/data/email-Eu-core-temporal.html)
- [SocioPatterns](https://sociopatterns.org/)




## Author

Dipendu Pal

MSc Student, IIT Bombay
