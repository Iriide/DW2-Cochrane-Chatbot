import numpy as np
import pandas as pd
import re
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import stopwords

UNCERTAIN_WORDS = {'maybe', 'possibly', 'might', 'could', 'not sure', 'uncertain', 'unknown', 'unsure', 'probably'}
STOPWORDS = set(stopwords.get_stopwords('en'))


def extract_keywords(text: str) -> set:
    # this function extracts keywords from the answer and removes the unneeded words, keeping the keywords
    return set(re.findall(r'\w+', text.lower())) - STOPWORDS


def has_uncertain_phrases_regex(text: str, threshold: int = 2) -> bool:
    text_lower = text.lower()
    count = 0
    for word in UNCERTAIN_WORDS:
        matches = re.findall(rf'\b{word}\b', text_lower)
        count += len(matches)
    return count >= threshold


def extract_section_numbers(text: str) -> set:
    matches = re.findall(r'§\s*(\d+)', text)
    return set(matches)


def get_valid_section_numbers(valid_titles: list[str]) -> set:
    section_numbers = set()
    for title in valid_titles:
        m = re.search(r'§\s*(\d+)', title)
        if m:
            section_numbers.add(m.group(1))
    return section_numbers


def has_invalid_sections(text: str, valid_titles: list[str]) -> bool:
    cited_numbers = extract_section_numbers(text)
    valid_numbers = get_valid_section_numbers(valid_titles)
    # Return True if any cited number is NOT in valid numbers
    return bool(cited_numbers - valid_numbers)


def is_low_quality_answer(answer: str, question: str, valid_titles: [], top_score: float, min_tokens: int = 15) -> (
        bool, dict):
    low_quality_answer_reason = []
    answer_words = answer.strip().split()
    # well if it's empty
    if not answer_words or len(answer_words) <= 1:
        low_quality_answer_reason.append('empty_answers')
        return True, low_quality_answer_reason
    # if the answer is same as the question
    if answer.strip().lower() == question.strip().lower():
        low_quality_answer_reason.append('same_as_question')
        return True, low_quality_answer_reason
    # if the answer was based on poorly suited passages
    if top_score < 0.55:
        low_quality_answer_reason.append('poor_passages')
        return True, low_quality_answer_reason
    # if less than 15 tokens
    if len(answer_words) < min_tokens:
        low_quality_answer_reason.append('too_short')
        return True, low_quality_answer_reason
    # if the answer provided by model does not have keywords from the question
    if not extract_keywords(question).intersection(extract_keywords(answer)):
        low_quality_answer_reason.append('no_keywords')
        return True, low_quality_answer_reason
    if has_invalid_sections(answer, valid_titles):
        low_quality_answer_reason.append('invalid_sections')
        return True, low_quality_answer_reason
    if has_uncertain_phrases_regex(answer):
        low_quality_answer_reason.append('uncertain_phrases')
        return True, low_quality_answer_reason
    return False, []


import torch

VALID = {"CORRECT", "INCORRECT", "NOT_ANSWERING_QUESTION"}


def llm_score_answer(model_answer: str, true_answer: str, model, tokenizer) -> str | None:
    device = next(model.parameters()).device

    prompt = (
        "You are a strict grader.\n"
        "Decide whether the MODEL ANSWER correctly answers the QUESTION, "
        "using the TRUE ANSWER as reference.\n\n"
        f"MODEL ANSWER:\n{model_answer}\n\n"
        f"TRUE ANSWER:\n{true_answer}\n\n"
        "Return exactly one label:\n"
        "CORRECT\n"
        "INCORRECT\n"
        "NOT_ANSWERING_QUESTION\n"
        "Label:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode only new tokens
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().upper()

    # take first “word”
    label = text.split()[0] if text else None

    if label in VALID:
        return label

    # fallback: search any valid label inside output
    for v in VALID:
        if v in text:
            return v
    return None


def score_answer(answer: str, true_answer: str, valid_titles: [], top_score: float, model, tokenizer) -> Dict[str, Any]:
    if not true_answer:
        return {
            'score': 0,
            'keyword_matches': 0,
            'length': 0,
            'uncertain_count': 0,
            'is_low_quality': False,
            'low_quality_answer_reason': [],
            'llm_score': None,
        }
    answer_clean = answer.strip().lower()
    answer_words = extract_keywords(answer_clean)
    question_keywords = extract_keywords(true_answer)

    keyword_matches = len(question_keywords & answer_words)
    length = len(answer_clean.split())
    uncertain_count = sum(w in answer_clean for w in UNCERTAIN_WORDS)
    # this is calculated +2 for kewords matching || +1 for length (capped at 30) || -2 for uncertain words
    score = keyword_matches * 2 + min(length, 30) - uncertain_count * 2

    llm_score = llm_score_answer(answer, true_answer, model, tokenizer)

    return {
        'score': score,
        'keyword_matches': keyword_matches,
        'length': length,
        'uncertain_count': uncertain_count,
        'is_low_quality': is_low_quality_answer(answer, true_answer, valid_titles, top_score)[0],
        'low_quality_answer_reason': is_low_quality_answer(answer, true_answer, valid_titles, top_score)[1],
        'llm_score': llm_score,
    }


def compute_overlap_percentage(answer: str, sources: List[str]) -> float:
    answer_tokens = set(re.findall(r'\w+', answer.lower()))
    source_tokens = set()
    for passage in sources:
        source_tokens.update(re.findall(r'\w+', passage.lower()))

    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & source_tokens
    return len(overlap) / len(answer_tokens)


def compute_similarity_with_sources(answer: str, sources: List[str], embed_fn) -> float:
    answer_embedding = embed_fn("Answer", answer)
    source_embeddings = [embed_fn("Source", s) for s in sources]
    dot_scores = [np.dot(answer_embedding, emb) for emb in source_embeddings]
    return float(np.mean(dot_scores)) if dot_scores else 0.0


def compute_trust_score(answer: str, sources: List[str], embed_fn) -> Dict[str, float]:
    overlap_pct = compute_overlap_percentage(answer, sources)
    similarity = compute_similarity_with_sources(answer, sources, embed_fn)
    return {
        "overlap_pct": overlap_pct,
        "semantic_similarity": similarity,
        "chunks_used": len(sources)
    }


class RAGModelSingleton:
    _instance = None

    @staticmethod
    def get_instance(*args, **kwargs):
        if RAGModelSingleton._instance is None:
            RAGModelSingleton._instance = RAGModel(*args, **kwargs)
        return RAGModelSingleton._instance


class RAGModel:

    def __init__(
            self,
            GENERATION_MODEL: str = "allenai/Olmo-3-7B-Instruct",
            EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5",
            VALIDATION_MODEL: str = "google/gemma-3-1b-it",
            DEVICE: str = None,
            read_from_file: bool = True,
    ):
        if DEVICE is None:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEVICE = DEVICE

        print(f"Using device: {self.DEVICE}")

        self.EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.EMB_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL).to(self.DEVICE).eval()

        self.GENERATION_MODEL = GENERATION_MODEL
        self.READ_FROM_FILE = read_from_file
        self.EMBEDDING_MODEL = EMBEDDING_MODEL

        self.TOKENIZER = AutoTokenizer.from_pretrained(self.GENERATION_MODEL, trust_remote_code=True)
        if self.TOKENIZER.pad_token is None:
            self.TOKENIZER.pad_token = self.TOKENIZER.eos_token

        self.OLMO_MODEL = (
            AutoModelForCausalLM.from_pretrained(
                self.GENERATION_MODEL,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            .to(self.DEVICE)
            .eval()
        )

        self.VALIDATION_TOKENIZER = AutoTokenizer.from_pretrained(
            VALIDATION_MODEL,
            trust_remote_code=True
        )
        self.VALIDATION_MODEL = (
            AutoModelForCausalLM.from_pretrained(
                VALIDATION_MODEL,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            .to(self.DEVICE)
            .eval()
        )

        first_param_device = next(self.OLMO_MODEL.parameters()).device
        print(f"Loaded model: {self.GENERATION_MODEL} on {first_param_device}")

        # load paragraphs
        self.df = pd.read_csv("src/data/all_postprocessed_paragraphs.csv")
        print(f"Loaded {len(self.df)} paragraphs for retrieval.")
        # build or load embeddings
        if not self.READ_FROM_FILE:
            print("Generating embeddings for documents...")
            self.df["Embedding"] = [
                self.embed_document(row.section, row.paragraph)
                for row in self.df.itertuples(index=False)
            ]
            print("Saving embeddings to embeddings.pkl...")
            self.df.to_pickle("embeddings.pkl")
        else:
            self.df = pd.read_pickle("embeddings.pkl")
            print("Loaded embeddings from embeddings.pkl.")

    def _embed_text(self, text: str, max_length: int = 1024) -> np.ndarray:
        """Turn text into a single embedding vector using EMB_MODEL hidden states (mean pooling)."""

        max_len = int(getattr(self.EMB_MODEL.config, "max_position_embeddings", max_length))
        max_len = min(max_len, max_length)

        inputs = self.EMB_TOKENIZER(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(self.DEVICE)

        with torch.no_grad():
            outputs = self.EMB_MODEL(**inputs)
            last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)
            pooled = last_hidden.mean(dim=1).squeeze(0)  # mean pooling

        return pooled.cpu().numpy()

    def embed_document(self, title: str, content: str) -> np.ndarray:
        text = f"{title}\n\n{content}" if title else content
        return self._embed_text(text)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_text(query)

    def get_tokens_used(self, text: str) -> Dict[str, float]:
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        keywords = len(extract_keywords(text))
        return {
            "words": word_count * 1.5,
            "characters": char_count // 3,
            "keywords": keywords * 3,
        }

    # ---------- RAG query ----------

    def ask(self, query: str, true_answer: str = None) -> Any:
        """Generate a response to the query using OLMO, with retrieval + scoring."""

        question_embedding = self.embed_query(query)  # shape (d,)
        print(question_embedding.shape)

        doc_embeddings = np.stack(self.df["Embedding"].to_list())  # (n_docs, d)
        dot_products = np.dot(doc_embeddings, question_embedding)  # (n_docs,)
        sorted_indices = np.argsort(dot_products)[::-1]

        delta_cutoff_ratio = 0.95  # Allow 5% drop from top score
        top_score = float(dot_products[sorted_indices[0]])
        print(top_score)

        top_indices = []
        for idx in sorted_indices:
            score = float(dot_products[idx])
            if score < top_score * delta_cutoff_ratio:
                break
            top_indices.append(idx)

        top_indices = top_indices[:5]
        top_passages_info = self.df.iloc[top_indices][["section", "paragraph"]]

        formatted_passages = [
            f"({row.section}) {row.paragraph}" for _, row in top_passages_info.iterrows()
        ]

        PROMPT = f"""You are a helpful and informative bot that answers questions using the reference passages below.
Use only the relevant information, and always cite the section title when answering (e.g., "According to Chapter..."). If the answer is not contained within the passages, do not assist with that question.

QUESTION: {query}

PASSAGES:
{chr(10).join(f'- {p}' for p in formatted_passages)}

ANSWER:"""

        PROMPT_SIMPLIFIED = f"""You are a helpful and informative bot that answers questions using the reference passages below.
Cite section titles in your answer for transparency.

QUESTION: {query}

PASSAGES:
{chr(10).join(f'- {row.section}' for _, row in top_passages_info.iterrows())}

ANSWER:"""

        inputs = self.TOKENIZER(
            PROMPT,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.DEVICE)

        print("prompt chars:", len(PROMPT))
        print("prompt tokens:", inputs["input_ids"].shape[1])

        tail = self.TOKENIZER.decode(
            inputs["input_ids"][0][-200:],
            skip_special_tokens=False
        )

        # --- generate ---
        with torch.no_grad():
            output_ids = self.OLMO_MODEL.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.TOKENIZER.pad_token_id,
                eos_token_id=self.TOKENIZER.eos_token_id,
            )

        # --- decode ONLY newly generated tokens ---
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0, prompt_len:]
        answer_text = self.TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()

        # optional cleanup for some chat/instruct models
        lower = answer_text.lower()
        if lower.startswith("assistant"):
            answer_text = answer_text[len("assistant"):].strip()
        if answer_text.lower().startswith("response:"):
            answer_text = answer_text[len("response:"):].strip()

        print("ANSWER_TEXT:\n", answer_text)

        custom_tokens_used = self.get_tokens_used(PROMPT + answer_text)
        total_output_tokens = int(output_ids.shape[1])
        tokens_used = total_output_tokens

        quality = score_answer(
            answer_text,
            true_answer,
            self.df["section"].tolist(),
            top_score,
            self.VALIDATION_MODEL,
            self.VALIDATION_TOKENIZER
        )

        source_texts = top_passages_info["paragraph"].tolist()
        trust = compute_trust_score(answer_text, source_texts, self.embed_document)

        used_chunks = [
            {"section": row.section, "content": row.paragraph}
            for _, row in top_passages_info.iterrows()
        ]

        return (
            answer_text,
            PROMPT_SIMPLIFIED,
            used_chunks,
            tokens_used,
            custom_tokens_used,
            quality,
            trust,
        )
