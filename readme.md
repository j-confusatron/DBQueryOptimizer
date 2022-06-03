# DB Query Plan Optimizer
Traditional database query planning uses dynamic programming to build cardinality estimations from the bottom up for query plan candidates. Cardinality estimates are just that, an estimate. Further, dynamic programming algorithms, such as Selinger's, break down as query complexity increases, requiring the databse engine to instead employ heuristics to make educated guesses.

This project introduces an intelligent system that learns to predict the optimal query plan to execute. Two pipelines are used:
- A production pipeline integrates tightly with the database to estimate cost for candidate plans and select the plan associated with the lowest cost estimate. All observed stateful data is recorded in a training buffer.
- A training pipeline pulls samples from the training buffer and further trains a neural network to accurately predict costs. Periodically, the current best model produced by the training pipeline is deployed to production.

By using reinforcement learning techniques, the system is able to produce its own sample data, from which it trains its self. The data produced is abstracted from specific schema details, such that the models produced may generalize their performance to unseen environments.

## Build Information

### Build Environment
- Development: Windows 11 / Anaconda 4.10.3 / Python 3.8.13
- Python Server + Database: Ubuntu 20.04.4 / Postgres 12.10 / Anaconda 4.10.3 / Python 3.8.13

### Libraries / Components
- Postgres 12.x
- gcc + make
- Python 3.8
- Pytorch
- Numpy
- Matplotlib
- Pandas
- Tqdm

## Postgres Function Hook
This project relies on prior work done on [Bao](https://dl.acm.org/doi/10.1145/3448016.3452838) (Marcus, et al.) to integrate with Postgres. This is done by building a function hook, that intercepts query plan calls. Instructions on building and deploying the query hook are [here](https://rmarcus.info/bao_docs/tutorial/1_pg_setup.html).

### Datasets
This system was trained using a variation of the IMDB JOB dataset. The production pipeline will produce samples for any database with traffic.