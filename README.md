# FinSearch: Intent-Aware Financial Document Intelligence

An end-to-end retrieval-augmented generation (RAG) system for financial documents. FinSearch routes user queries by intent, retrieves relevant passages from 41 financial PDFs across 4 regulatory categories, reranks with an LLM, generates a grounded answer, and scores it for faithfulness — all without hallucination.

---

## Project Structure

```
FinSearch/
├── baseline/                   # Week 1 — BM25 Baseline
├── dense_retrieval/            # Week 2 — Dense Retrieval (MiniLM FAISS)
├── hybrid/                     # Week 3 — Hybrid Retrieval (BM25 + Dense)
├── finrerank/                  # Week 4 — LLM Reranking + Full Pipeline
├── pdf_chunking/               # Week 0 — Chunking Strategy Evaluation
├── intent_classification/      # Week 5 — Intent Classifier
├── fine_tuning/                # Week 6 — MiniLM Fine-Tuning (in progress)
├── Dataset/                    # FiQA corpus, queries, qrels + PDF knowledge base
├── config.py                   # Central path config
├── requirements.txt
└── poster_plots.py             # Generates all comparison charts → poster_images/
```

---

## Knowledge Base

41 financial PDFs across 4 categories:

| Category | Description |
|----------|-------------|
| `Regulatory` | Central bank and securities regulation documents |
| `Consumer_Protection` | Financial consumer protection guidelines |
| `Payment_Industry` | Payment systems, card network standards |
| `Synthetic_Policies` | Complaint procedures and internal policy documents |

---

## Experiments & Results

### Week 0 — PDF Chunking Strategy Evaluation

**Goal:** Find the best way to split financial PDFs into retrieval-ready chunks.  
**Corpus:** 4 representative PDFs (1 per category), 20 synthetic QA pairs.  
**Files:** `pdf_chunking/PDF_Chunking.ipynb`

5 strategies evaluated with MiniLM and BGE-Large:

| Strategy | Description | MiniLM Recall@10 | BGE-Large Recall@10 |
|----------|-------------|:---:|:---:|
| S1 | Sliding window, 512 words | 0.35 | 0.35 |
| S2 | Sliding window, 256 words | 0.40 | 0.40 |
| S3 | Paragraph-based | 0.30 | 0.35 |
| **S4** | **Token-Exact 200/400 tokens** | **0.75** | **0.80 ✓** |
| S5 | Section-aware 200/400 | 0.60 | 0.75 |

**Winner: S4 Token-Exact (400 tokens, 100 overlap) with BGE-Large — Recall@10 = 0.80**

> Token-exact chunking uses a HuggingFace tokenizer to split on exact token boundaries (zero truncation), which outperforms word-count methods on formal legal/regulatory language.

---

### Week 1 — BM25 Baseline

**Goal:** Establish a keyword-search baseline on FiQA financial QA dataset.  
**Files:** `baseline/Baseline_model.ipynb`

| Model | NDCG@10 | MRR | Recall@10 | Queries |
|-------|:-------:|:---:|:---------:|:-------:|
| BM25 (k1=1.2, b=0.75) | 0.2169 | 0.2706 | 0.2784 | 648 |

BM25 struggles with synonym mismatch — e.g., user says "returns", document says "yield".

---

### Week 2 — Dense Retrieval

**Goal:** Replace keyword search with semantic vector search.  
**Files:** `dense_retrieval/Dense_Retrieval.ipynb`

| Model | NDCG@10 | MRR | Recall@10 | vs BM25 |
|-------|:-------:|:---:|:---------:|:-------:|
| BM25 Baseline | 0.2169 | 0.2706 | 0.2784 | — |
| **MiniLM-L6-v2 FAISS** | **0.3687** | **0.4451** | **0.4413** | **+70%** |

Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), FAISS `IndexFlatIP` with cosine similarity, full FiQA corpus (57K passages).

---

### Week 3 — Hybrid Retrieval

**Goal:** Combine BM25 (lexical) + Dense (semantic) for best of both.  
**Files:** `hybrid/Hybrid_RRF.ipynb`

Two fusion strategies tested — alpha-weighted interpolation and Reciprocal Rank Fusion (RRF):

| Model | NDCG@10 | MRR | Recall@10 |
|-------|:-------:|:---:|:---------:|
| BM25 Baseline | 0.2169 | 0.2706 | 0.2784 |
| MiniLM Dense | 0.3687 | 0.4451 | 0.4413 |
| Hybrid RRF (k=60) | 0.3519 | 0.4171 | 0.4396 |
| **Hybrid Alpha (α=0.7)** | **0.3791** | **0.4606** | **0.4473** |

**Alpha sweep result:** α=0.7 (70% dense, 30% BM25) is the sweet spot.

| α | NDCG@10 |
|---|:-------:|
| 0.1 | 0.2656 |
| 0.3 | 0.3084 |
| 0.5 | 0.3593 |
| **0.7** | **0.3791** |
| 0.9 | 0.3735 |

---

### Week 4 — LLM Reranking + Full End-to-End Pipeline

**Goal:** Add Query Expansion, LLM reranking, LLM answer generation, and NLI confidence scoring.  
**Files:** `finrerank/FinChatbot.ipynb`, `finrerank/FinPipeline_Comparison.ipynb`

#### Pipeline Architecture (B1 — Best)

```
User Query
    │
    ▼
[1] Query Expansion ─────── Groq LLaMA 3.3 70B
    │  Append 8-12 financial synonyms/related terms to query
    │
    ▼
[2] Dense Retrieval ──────── BGE-Large-EN-v1.5 (1024-dim) + FAISS
    │  Retrieve Top-50 candidates from stratified corpus
    │
    ▼
[3] LLM Reranking ────────── Mistral Large 2411 (via OpenRouter)
    │  Score top-20 passages → return ranked JSON → take Top-10
    │
    ▼
[4] Answer Generation ────── LLaMA 3.3 70B (via OpenRouter)
    │  "Answer using ONLY the provided documents"
    │
    ▼
[5] Confidence Scoring ───── DeBERTa-v3-small NLI CrossEncoder
       Retrieval conf  = mean(normalized retrieval scores)        → weight: 40%
       Faithfulness    = mean(NLI entailment(doc, answer))        → weight: 60%
       Final score     = HIGH (≥0.7) / MED (≥0.4) / LOW (<0.4)
```

#### 4-Pipeline Comparison (194 queries, stratified FiQA sub-corpus)

| Pipeline | Dense Model | Retrieval | NDCG@10 | MRR | Recall@10 |
|----------|-------------|-----------|:-------:|:---:|:---------:|
| A1 | MiniLM (384-dim) | Dense only | 0.5917 | 0.6607 | 0.6724 |
| A2 | MiniLM (384-dim) | Hybrid α=0.7 | 0.5813 | 0.6685 | 0.6513 |
| **B1** | **BGE-Large (1024-dim)** | **Dense only** | **0.6056** | **0.6679** | **0.6917** |
| B2 | BGE-Large (1024-dim) | Hybrid α=0.7 | 0.5381 | 0.6243 | 0.5984 |

**Winner: B1 — BGE-Large Dense + QE + Mistral Rerank + LLaMA Answer**

> All 4 pipelines use the same QE (Groq LLaMA 3.3 70B), same reranker (Mistral Large), same answer model (LLaMA 3.3 70B), and same NLI confidence scorer (DeBERTa-v3-small). Only the retrieval encoder differs.

#### Full Progression (all weeks combined)

| Stage | Model | NDCG@10 | MRR | Recall@10 |
|-------|-------|:-------:|:---:|:---------:|
| Week 1 | BM25 Baseline | 0.2169 | 0.2706 | 0.2784 |
| Week 2 | MiniLM Dense | 0.3687 | 0.4451 | 0.4413 |
| Week 3 | Hybrid α=0.7 | 0.3791 | 0.4606 | 0.4473 |
| Week 4 | Hybrid + Mistral Rerank | 0.3885 | 0.4775 | 0.4485 |
| **Week 4** | **B1 Full Pipeline** | **0.6056** | **0.6679** | **0.6917** |

> NDCG@10 improvement from BM25 to B1: **+179%**

---

### Week 5 — Intent Classification

**Goal:** Route user queries to the correct knowledge-base category before retrieval.  
**Files:** `intent_classification/FinIntent_Classifier.ipynb`, `intent_classification/FinIntent_DataPrep.ipynb`

3 classifiers compared on 4 categories (Regulatory, Consumer Protection, Payment Industry, Synthetic Policies):

| Model | Training Data | Banking77 Acc | QA Eval Acc (120 q) |
|-------|--------------|:-------------:|:-------------------:|
| Zero-Shot DeBERTa NLI | None | 25.5% | 5.0% |
| Fine-Tuned MiniLM (PDF-Only) | 600 Groq PDF questions | 73.5% | 75.8% |
| **Fine-Tuned MiniLM (Full)** | **Banking77 + Groq PDF** | **93.0%** | **90.0%** |

**Winner: Fine-Tuned MiniLM on full dataset**

- Training: Banking77 (~10K conversational finance) + ~2,400 Groq-generated PDF-domain questions
- Evaluation: 120 held-out Groq questions (30 per category) — same distribution as training
- Saved to: `intent_classification/minilm_intent_classifier/`
- Load: `AutoModelForSequenceClassification.from_pretrained('intent_classification/minilm_intent_classifier')`

> Training on diverse data (Banking77 + PDF domain) outperforms PDF-only (75.8%) because the conversational Banking77 style generalizes intent patterns across domains.

---

### Week 6 — MiniLM Retrieval Fine-Tuning (In Progress)

**Goal:** Fine-tune `all-MiniLM-L6-v2` on domain-specific (question, chunk) pairs from the 41 PDFs.  
**Files:** `fine_tuning/FinS4_BuildCorpus.ipynb`, `fine_tuning/FinMiniLM_FineTune.ipynb`, `fine_tuning/FinMiniLM_Eval.ipynb`

| Step | File | Output |
|------|------|--------|
| 1. Build S4 corpus | `FinS4_BuildCorpus.ipynb` | `fine_tuning/s4_corpus.csv` (S4 chunks from all 41 PDFs) |
| 2. Map questions to chunks | `FinS4_BuildCorpus.ipynb` | `fine_tuning/training_pairs.csv` (10,186 pairs, avg sim=0.52) |
| 3. Fine-tune | `FinMiniLM_FineTune.ipynb` | `fine_tuning/minilm_finetuned/` |
| 4. Evaluate | `FinMiniLM_Eval.ipynb` | Recall@10 + MRR vs base MiniLM |

Training: `MultipleNegativesRankingLoss`, 3 epochs, batch size 32, ~2,400 training pairs.

> Current training pairs are similarity-mapped (question → nearest chunk via cosine sim), which introduces noise. Industry best practice is to generate questions directly from chunks (chunk → LLM → question) with `chunk_id` tracking to guarantee true positive pairs.

---

## Models Used Across the Project

| Role | Model | Where |
|------|-------|--------|
| Dense retrieval (baseline) | `sentence-transformers/all-MiniLM-L6-v2` | Local |
| Dense retrieval (best) | `BAAI/bge-large-en-v1.5` | Local |
| Query expansion | `meta-llama/llama-3.3-70b-instruct` | OpenRouter API |
| LLM reranker | `mistralai/mistral-large-2411` | OpenRouter API |
| Answer generation | `meta-llama/llama-3.3-70b-instruct` | OpenRouter API |
| NLI confidence scorer | `cross-encoder/nli-deberta-v3-small` | Local |
| Intent classifier | Fine-tuned `all-MiniLM-L6-v2` | Local (saved) |
| Question generation | `llama3-8b-8192` | Groq API |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys
Create a `.env` file in the repo root:
```
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### 3. Run in order

| Step | Notebook | Purpose |
|------|----------|---------|
| 0 | `pdf_chunking/PDF_Chunking.ipynb` | Chunking strategy evaluation |
| 1 | `baseline/Baseline_model.ipynb` | BM25 baseline |
| 2 | `dense_retrieval/Dense_Retrieval.ipynb` | Dense FAISS retrieval |
| 3 | `hybrid/Hybrid_RRF.ipynb` | Hybrid BM25 + Dense |
| 4 | `finrerank/FinChatbot.ipynb` | LLM rerank + answer + confidence |
| 4b | `finrerank/FinPipeline_Comparison.ipynb` | Compare all 4 pipelines |
| 5a | `intent_classification/FinIntent_DataPrep.ipynb` | Generate training data |
| 5b | `intent_classification/FinIntent_Classifier.ipynb` | Train intent classifier |
| 6a | `fine_tuning/FinS4_BuildCorpus.ipynb` | Build S4 corpus + training pairs |
| 6b | `fine_tuning/FinMiniLM_FineTune.ipynb` | Fine-tune MiniLM retriever |
| 6c | `fine_tuning/FinMiniLM_Eval.ipynb` | Evaluate base vs fine-tuned |

### 4. Generate poster comparison charts
```bash
python3 poster_plots.py
# Output: poster_images/  (4 comparison figures)
```

---

## Key Findings

1. **Token-exact chunking (S4) is critical** — Recall@10 jumps from 0.35–0.40 (word-based) to 0.80 (token-exact) on financial PDFs.
2. **Dense beats BM25 by 70%** — semantic understanding handles finance vocabulary mismatch.
3. **Hybrid (α=0.7) adds marginal gain** — dense already captures most semantics; BM25 helps on exact term matches.
4. **BGE-Large outperforms MiniLM** — 1024-dim vs 384-dim matters for domain-specific formal text.
5. **Query expansion is high-ROI** — prepending LLM-generated financial synonyms before encoding boosts retrieval without any retraining.
6. **Mistral reranking closes the gap** — precision-at-top improves significantly even when base recall is similar.
7. **Intent classification needs diverse training** — PDF-only fine-tuning (75.8%) is significantly worse than mixing Banking77 + PDF questions (90%).

---

## Dataset

- **FiQA** (Financial Question Answering): 57K passages, 648 test queries with relevance judgements — used for retrieval evaluation.
- **Banking77**: 10K labeled banking intent queries across 77 classes — mapped to 4 categories for intent classifier training.
- **41 Financial PDFs**: Internal knowledge base across 4 regulatory/industry categories — chunked with S4 strategy for production retrieval.
- **Groq-generated questions**: ~2,400 PDF-domain questions generated by LLaMA-3 from the 41 PDFs, used for intent classifier training and retrieval fine-tuning.
