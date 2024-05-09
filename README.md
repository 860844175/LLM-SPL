# LLM-SPL

## Data Processnig(`preprocessing/`)
* Collect CVE and commit information from NVD and GitHub repositories (`collect_cve_data.py`, `get_coe_diff.py`).
* Build dataset (`create_dataset.py`) and extract textual features (F0) (`feature.py`) following the "VCMatch" method.
* Construct the dataset and extract textual information for feature engineering


## Commit Graph Construction(`graph_construction/`)
* Matching commit-commit relationships using information from the dataset and download GitHub repositories. 
* Construct a commit-commit relationship graph based on the matched relationships
* Leverage metadata and extracted textual information from GitHub to build the graph


## LLM Integration(`acquire_LLM_feedback/`)
* Utilize LLMs (e.g., GPT-3.5) to 
    * provide the relevance feedback between CVE and commit (F1) to enhance its representation (`llm_relevance_feedback.py`)
    * provide guidance for creating inter-commit relationship graph(F2). (`llm_inter_commit_feedback.py`)


## Model(`model/')
* Baseline Model(`Base_SPL.py`)
    * we utilize the VCMatch as our baseline model to provide initial feedback through inputing two feature set F0, F0 + F1.

* Main Model:
    * Input the LLM-enhanced CVE-commit feature set (F0 + F1) and the inter-commit relationship graph (F2) into the model for training(`run.py`)

## Features
* F0: Textual features extracted from CVE descriptions and commit messages using the "VCMatch" method
* F1: Relevance feedback between CVEs and commits provided by LLMs
* F2: Inter-commit relationship graph constructed using metadata and LLM guidance