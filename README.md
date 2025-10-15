# Predicting Recall@k for Large-Scale Dense Information Retrieval via Score Distribution Modeling

[![Paper](https://img.shields.io/badge/Paper-ECIR%202026-blue)](TODO)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the source code and experiments for the paper **Modeling Score Distributions for Large-Scale Dense Information Retrieval**.

---

## 🧩 Overview

Dense Information Retrieval (DIR) has shown strong performance on standard-scale benchmarks such as MS MARCO, but its behavior at **large corpus sizes** remains insufficiently understood.
This work introduces the **Score Distribution Model (SDM)** — a probabilistic framework that models **query-document similarity scores** as skew-normal (and hybrid) distributions, allowing us to **predict retrieval performance (Recall@k)** for much larger indices than those directly observed.

Using the **CoRE benchmark** (Controlled Retrieval Evaluation), we show that SDM can accurately forecast recall degradation when scaling from **10k to 100M documents**, providing a theoretical and practical tool for analyzing large-scale dense retrieval systems.

<p align="center">
  <img src="images/overview.png" width="60%">
</p>

---

## 📘 Abstract

> Understanding how dense retrievers behave at large index sizes remains a central challenge in information retrieval.
> We introduce the **Score Distribution Model (SDM)**, which models similarity scores between queries and relevant/non-relevant documents as approximate normal distributions.
> From these distributions, we derive a closed-form expression to compute **Recall@k** across increasing corpus sizes based on scores observed in a smaller base corpus.
> Experiments on the **CoRE benchmark** show that SDM can predict retrieval performance for corpora up to 100M passages from statistics estimated on only 10k documents.

---

## 🧠 Method

The method models the **score distributions** between queries and documents as:

- Relevant scores:  
  $S_r \sim \text{SN}(\xi_r, \omega_r, \alpha_r)$

- Non-relevant scores:  
  $S_n$ is modeled as a **hybrid** of a skew-normal body and a **Generalized Pareto Distribution (GPD)** tail.

The Recall@k is computed by solving the following equation:

$$
R \cdot \alpha_r(\tau_k) + N \cdot \alpha_n(\tau_k) = k
$$

with

$$
\alpha_r(\tau) = 1 - F_r(\tau) \qquad \text{and} \qquad \alpha_n(\tau) = 1 - F_n(\tau)
$$

where $F_r$ and $F_n$ denote the CDFs of the relevant and non-relevant score distributions, respectively.

The cutoff score $\tau_k$ is the score of the k-th highest scoring document in the entire corpus. The Recall@k can then be computed as:

$$
\text{Recall@k} = 1 - F_r(\tau_k)
$$

---

## 📊 Experiments

Experiments are conducted on the **CoRE benchmark**, which provides passage and document retrieval tasks across corpus sizes from **10k to 100M**.

**Key Results:**

* Predictions based on the 10k subset match observed Recall@k values up to 100M passages within **≈2% deviation**.
* Demonstrates a quantifiable **trade-off** between corpus size and ranking depth (k).
* Validated on multiple dense retrievers:

  * [Snowflake Arctic M v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)
  * [Snowflake Arctic M v1](https://huggingface.co/Snowflake/snowflake-arctic-embed-m)
  * [Jina v3](https://huggingface.co/jinaai/jina-embeddings-v3)
  * [Multilingual E5 Large Instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/padas-lab-de/ecir26-score-distribution-model.git
cd ecir26-score-distribution-model
```

There are two ways you can install the dependencies to run the code.

### Using Poetry (recommended)

If you have the [Poetry](https://python-poetry.org/) package manager for Python installed already, you can simply set up everything with:

```bash
poetry install
source $(poetry env info --path)/bin/activate
```

After the installation of all dependencies, you will end up in a new shell with a loaded venv. In this shell, you can run the main `sdm` command. You can exit the shell at any time with `exit`.

```bash
sdm --help
```

To install new dependencies in an existing poetry environment, you can run the following commands with the shell environment being activated:

```bash
poetry lock
poetry install
```

### Using Pip (alternative)

You can also create a venv yourself and use `pip` to install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

---

## 🚀 Usage

### 1️⃣ Compute Score Distributions

> [!NOTE]
> The first time you run this command, it might take a while as it loads the entire CoRE dataset collection.

```bash
# Models: snowflakev2, snowflake, jina3, e5
# Examples:
sdm compute-score-distributions snowflakev2 passage

# To run for all models and both passage/document collections:
sdm compute-score-distributions
```

### 2️⃣ Visualize Score Distributions

```bash
sdm visualize distribution
```

### 3️⃣ Predict Recall@k for Larger Corpora

In order to have a reference for the predicted results, we first compute the empirical results on the larger corpora.

```bash
sdm compute-empirical-results
```

Then, we can visualize the empirical vs predicted Recall@k curves.

```bash
sdm visualize prediction
```

---

## 📦 Repository Structure

```
.
├── src/sdm/cli/
│   ├── compute-score-distributions.py
│   ├── compute-empirical-results.py
│   └── visualize/
│       ├── distribution.py
│       └── prediction.py
├── resources/                 # Output figures (e.g. Recall@k curves)
├── images/                    # Example images for the README
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 📈 Example Visualization

Example output showing **empirical vs predicted Recall@k** curves for the CoRE passage collection (Dense retriever: [Snowflake Arctic M v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)):

<p align="center">
  <img src="images/recall.png" width="30%">
</p>

---

## ⚙️ Development

### Run Code Formatting

To run the code formatting, you can use the following command:

```bash
isort .
black .
```

---

<!-- ## 🧩 Citation

If you use this work, please cite:

```bibtex
@inproceedings{author2025sdm,
  title     = {Modeling Score Distributions for Large-Scale Dense Information Retrieval},
  author    = {Author, First and Author, Second and Author, Third},
  booktitle = {Proceedings of the European Conference on Information Retrieval (ECIR)},
  year      = {2025},
  note      = {Short Paper}
}
```

--- -->

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## 🔗 Related Work

* Reimers & Gurevych (2021): *The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes*
