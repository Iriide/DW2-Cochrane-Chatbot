# Retrieval-Augmented Generation (RAG) Model

This repository implements a **Retrieval-Augmented Generation (RAG)** model using **PyTorch** and **Hugging Face Transformers**. The system retrieves relevant passages from a structured corpus and generates answers to user queries grounded in the retrieved evidence.

The project is part of the **GRADE AI initiative**, which aims to develop trustworthy, domain-specific LLM-based chatbots for evidence-based medicine resources, including the *GRADE Handbook* and the *Cochrane Handbook for Systematic Reviews of Interventions*.

---

## Project Objectives

Building on the successful GRADE book chatbot, this project focuses on:

1. Selecting a specific chapter of the **Cochrane Handbook**.
2. Creating a curated set of questions with varying levels of difficulty, along with expert-validated answers.
3. Developing a chatbot capable of answering questions based on the selected chapter.
4. Evaluating the chatbot’s performance against the predefined question set.

---

## Key Features

- **Model Loading**  
  Supports loading pre-trained language models for answer generation and validation.

- **Document Embedding**  
  Converts handbook paragraphs into vector embeddings for efficient semantic retrieval.

- **Retrieval-Augmented Question Answering**  
  Retrieves the most relevant passages and generates answers grounded in those sources.

- **Customizable Prompting**  
  Flexible prompt templates for generation and evaluation tasks.

- **Scoring and Trust Evaluation**  
  Computes answer scores and trust metrics based on similarity between source and generated embeddings.

- **Singleton Model Architecture**  
  Ensures the model is initialized only once to reduce memory overhead.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Prepare data files

Ensure the following files are available:

- `src/data/all_postprocessed_paragraphs.csv`
Preprocessed text passages used for retrieval.

- `embeddings.pkl` (optional)
If not present, embeddings will be generated automatically during initialization.

## Usage
### Running the application
The project provides a Streamlit-based web interface for interactive use.
```bash
streamlit run App.py
```
Once launched, you can:

- Run benchmarks using predefined expert-curated questions.
- Ask custom questions related to the handbook content.

### Predefined Questions and Benchmarking

- Source \
Questions are derived from the Cochrane Handbook for Systematic Reviews of Interventions (version 6.3).

- Purpose \
Designed to evaluate the chatbot’s ability to answer questions across different levels of complexity.

- Validation \
All predefined answers have been reviewed and verified by human domain experts.

## Repository Structure
```
.
├── App.py                 # Streamlit application
├── RAGmodel.py            # Core RAG implementation
├── src/
│   ├── data/              # Retrieval dataset
│   └── ...                # Supporting modules
├── embeddings.pkl         # Cached embeddings (optional)
├── requirements.txt
└── README.md
```

## Requirements
Requirements
* Python 3.8 or higher
* PyTorch
* Hugging Face Transformers
* NumPy
* Pandas
Note: CUDA support is recommended for faster inference but is not mandatory.