# Universal Document Summarizer (BART + RAG + Verifier)

**Date:** 2025-08-16  
**Authors:** _Add names, emails_  
**Repository:** https://github.com/USERNAME/UniversalSummarizer _(replace with real link)_

---

## 1. Introduction
**Problem.** We target faithful summarization of long, unstructured documents (TXT/PDF/DOCX/HTML). Users need quick,
reliable digests and optional quality checks without pre‑curated datasets.

**Approach.** We built a Streamlit app that combines:
- **Abstractive summarization** with `facebook/bart-large-cnn`,
- An **extractive** TextRank baseline for comparison,
- Optional **RAG** (Sentence-Transformers + FAISS) to retrieve salient passages for BART,
- A lightweight **factuality check** using NLI (`microsoft/deberta-large-mnli`) that estimates sentence‑level entailment.

We expose four pipelines: _BART_, _Extractive (TextRank)_, _RAG + BART_, and variants _with Verifier_.  
The code and UI are in a single `app.py` for easy deployment.

---

## 2. Data
**Inputs.** The app accepts user‑provided documents: `.txt`, `.pdf`, `.docx`, and URLs (HTML). There is no fixed training set.

**Preprocessing.**
- **PDF/DOCX** text extraction via `PyPDF2` / `python-docx`.
- **HTML** cleaned with BeautifulSoup (scripts/styles/nav removed).
- **Chunking**: long texts are split into ~900 token (~3600 char) chunks; summaries are concatenated and optionally re‑compressed.
- **RAG**: texts are split into ~180‑word passages; MiniLM embeddings + FAISS index enable top‑k retrieval.

**Challenges.**
- PDFs with complex layout/tables can yield noisy text.
- Very long inputs exceed model limits; chunking compresses content but may lose cross‑chunk context.
- Web pages may include repeated boilerplate; more robust cleaners could help.

> _If you evaluated on a specific dataset (e.g., news, academic papers), describe it here with size, domain, and any filters you applied._

---

## 3. Models & Methods
- **BART (abstractive)**: `facebook/bart-large-cnn` via `transformers.pipeline`. Dynamic length with _auto_ heuristic; presets for tweet/short/medium/long.
- **Extractive baseline**: TextRank via `sumy` (fallback keyword scoring if `sumy` unavailable).
- **RAG**: `all-MiniLM-L6-v2` embeddings; FAISS inner‑product index over 180‑word passages. For generic summarization we query with the document head; top‑k passages are concatenated before BART.
- **Factuality (NLI)**: `microsoft/deberta-large-mnli`; we score each summary sentence vs. a premise window (~3000 chars) and report average entailment. Low‑confidence sentences are flagged.

**Rationale.**
- BART‑CNN is a strong off‑the‑shelf summarizer for English news‑style text.
- RAG helps surface salient spans in long inputs.
- NLI provides a practical, model‑agnostic proxy for factual consistency.

---

## 4. Results & Evaluation
We report ROUGE‑1/2/L and BERTScore when a **reference** summary is available. Example protocol:
1. Produce summaries with each method (BART, TextRank, RAG+BART).
2. Compute ROUGE/BERTScore against references.
3. Record runtime and compression ratio.

**Template (fill in):**

| Method                | ROUGE‑1 F1 | ROUGE‑2 F1 | ROUGE‑L F1 | BERTScore F1 | Time (s) | Compression |
|-----------------------|-----------:|-----------:|-----------:|-------------:|---------:|------------:|
| BART (auto)           |            |            |            |              |          |             |
| Extractive TextRank   |            |            |            |              |          |             |
| RAG + BART            |            |            |            |              |          |             |
| BART + Verifier*      |     —      |     —      |     —      |      —       |          |             |

\*Verifier does not change the summary; it flags low‑entailment sentences and reports average entailment.

**Observations (example):**
- RAG tends to help on long, heterogeneous sources; minimal effect on short inputs.
- TextRank provides a fast baseline but may be less coherent.
- NLI flags sentences with numbers/dates—useful for manual review.

---

## 5. Contributions
- **You / Teammates:** _List who did what (app UI, BART integration, RAG, NLI, evaluation, docs, etc.)._
- **Found online:** Hugging Face model cards and examples; Streamlit docs; Sumy examples.
- **GenAI usage:** Used ChatGPT **only** for organizing the report/README, code review suggestions, and proofreading; **all implementation choices and data were our own**.

---

## 6. Challenges & Future Work
- **Speed & Memory:** NLI model is large; consider a smaller MNLI model or distillation, batching sentences, or GPU inference.
- **Retrieval Quality:** Replace head‑query with coverage‑based selection (e.g., MMR) or section/title‑aware queries.
- **Long Context:** Explore long‑context summarizers (e.g., Longformer‑Encoder‑Decoder, Llama‑Long) or sliding‑window with overlap.
- **Factuality:** Condition BART on retrieved evidence (cite spans) and run per‑sentence premise selection for NLI.
- **Quality Controls:** Add hallucination tests, date/number normalization, and domain‑specific evaluation sets.
- **Productization:** Add caching, API layer, auth, and telemetry; package as Docker for reproducible runs.

---

## 7. Improvements Since Previous Submission (if applicable)
_Explicitly list what changed: new methods, better chunking, RAG, verifier, UI improvements, new metrics, bug fixes._

---

## 8. Current State & Plans
- **Current state:** App runs end‑to‑end with BART, TextRank, optional RAG and NLI; evaluation tab available.
- **Future plans:** _Add items you intend to complete after the deadline; these will not be part of this submission._
