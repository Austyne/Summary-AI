"""
Full BART Document Summarizer - Streamlit App (Upgraded)
========================================================
- Methods: TextRank | BART | RAG + BART | BART + Verifier | RAG + BART + Verifier
- Factuality check with NLI (DeBERTa MNLI)
- Optional ROUGE / BERTScore evaluation if reference summary provided
"""

import os, io, re, time, warnings
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

# ---------- Core / Models ----------
from transformers import (
    BartTokenizer, BartForConditionalGeneration, pipeline,
    AutoTokenizer, AutoModelForSequenceClassification
)
from rouge_score import rouge_scorer
from bert_score import score as bertscore

# ---------- Document processing ----------
import PyPDF2
try:
    from docx import Document
except ImportError:
    Document = None
import requests
from bs4 import BeautifulSoup

# ---------- Extractive baseline ----------
try:
    from sumy.parsers.plaintext import PlainTextParser
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
    from sumy.summarizers.textrank import TextRankSummarizer
    HAS_SUMY = True
except Exception:
    HAS_SUMY = False

# ---------- RAG ----------
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_ST = True
except Exception:
    HAS_ST = False

# ---------- UI ----------
import streamlit as st

# ---------------- Streamlit page config ----------------
st.set_page_config(
    page_title="🚀 Universal Summarizer (BART + RAG + Verifier)",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
#                 Document Processor
# ======================================================
class UniversalDocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.md']

    def read_text_file(self, file_content) -> str:
        try:
            return str(file_content, 'utf-8')
        except UnicodeDecodeError:
            return str(file_content, 'latin-1')

    def read_pdf_file(self, file_content) -> str:
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                txt = page.extract_text() or ""
                text += txt + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    def read_docx_file(self, file_content) -> str:
        try:
            if Document is None:
                st.error("python-docx not installed. Install with: pip install python-docx")
                return ""
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            text = "\n".join(p.text for p in doc.paragraphs)
            return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""

    def read_url(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            for s in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                s.decompose()
            text = soup.get_text(separator="\n")
            lines = (l.strip() for l in text.splitlines())
            chunks = (p.strip() for l in lines for p in l.split("  "))
            return "\n".join(c for c in chunks if c)
        except Exception as e:
            st.error(f"Error reading URL: {e}")
            return ""

# ======================================================
#                 BART Loader + Summarizer
# ======================================================
@st.cache_resource(show_spinner=False)
def load_bart_model():
    model_name = "facebook/bart-large-cnn"
    tok = BartTokenizer.from_pretrained(model_name)
    mdl = BartForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    pipe = pipeline("summarization", model=mdl, tokenizer=tok, device=0 if device=="cuda" else -1)
    return {"pipeline": pipe, "tokenizer": tok, "model": mdl, "device": device}

class SmartBARTSummarizer:
    def __init__(self, bart_components):
        self.pipeline = bart_components['pipeline']
        self.tokenizer = bart_components['tokenizer']
        self.model = bart_components['model']
        self.device = bart_components['device']
        self.summary_presets = {
            'tweet': {'max_length': 50, 'min_length': 20},
            'short': {'max_length': 100, 'min_length': 40},
            'medium': {'max_length': 200, 'min_length': 80},
            'long': {'max_length': 400, 'min_length': 150},
            'auto': {'max_length': None, 'min_length': None}
        }

    def _smart_length(self, text: str) -> Dict[str, int]:
        wc = len(text.split())
        if wc < 100: ratio = .7
        elif wc < 500: ratio = .4
        elif wc < 1500: ratio = .25
        elif wc < 5000: ratio = .15
        else: ratio = .1
        target = max(30, int(wc*ratio))
        return {
            "max_length": min(400, int(target*1.3)),
            "min_length": max(20, int(target*.7)),
            "target_words": target,
            "compression_ratio": ratio
        }

    def _chunk_long_text(self, text: str, max_tokens: int = 900) -> List[str]:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return [text]
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        chunks, cur = [], ""
        for p in paragraphs:
            if len(cur)+len(p) <= max_chars:
                cur += p + "\n\n"
            else:
                if cur: chunks.append(cur.strip())
                cur = p + "\n\n"
        if cur: chunks.append(cur.strip())
        return chunks

    def summarize_bart(self, text: str, style: str = 'auto') -> str:
        if style == 'auto':
            length = self._smart_length(text)
        else:
            length = self.summary_presets[style].copy()
        chunks = self._chunk_long_text(text)
        outs = []
        for i, ch in enumerate(chunks):
            res = self.pipeline(
                ch,
                max_length=min(200, length['max_length']) if len(chunks)>1 else length['max_length'],
                min_length=max(20, (length['min_length'] or 20)),
                do_sample=False,
                truncation=True
            )
            outs.append(res[0]['summary_text'])
        final = " ".join(outs)
        # second pass if needed
        if len(outs) > 1 and len(final.split()) > (length['max_length'] or 200):
            res2 = self.pipeline(
                final,
                max_length=length['max_length'] or 200,
                min_length=length['min_length'] or 80,
                do_sample=False,
                truncation=True
            )
            final = res2[0]['summary_text']
        return final

# ======================================================
#                 Extractive Baseline (TextRank)
# ======================================================
def textrank_summary(text: str, sentence_count: int = 5) -> str:
    if HAS_SUMY:
        parser = PlainTextParser.from_string(text, SumyTokenizer("english"))
        summarizer = TextRankSummarizer()
        sents = summarizer(parser.document, sentence_count)
        return " ".join(str(s) for s in sents)
    # ---- Basit yedek (sumy yoksa): en çok anahtar kelime içeren cümleleri seç ----
    sentences = re.split(r'(?<=[.!?])\s+', text)
    words = re.findall(r"\b\w+\b", text.lower())
    stop = set("""the a an and or but if for to of in on at by with from this that is are was were be been being as it its it's into about over after before between among against during without within through per than then so such also can may might will would should could""".split())
    freq = {}
    for w in words:
        if w not in stop and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    scores = []
    for s in sentences:
        sw = re.findall(r"\b\w+\b", s.lower())
        score = sum(freq.get(w,0) for w in sw if w not in stop)
        scores.append((score, s))
    top = [s for _, s in sorted(scores, reverse=True)[:max(1, sentence_count)]]
    return " ".join(top)

# ======================================================
#                 RAG (Simple)
# ======================================================
class SimpleRAG:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embed_model)
        self.index = None
        self.passages = []

    def _split(self, text: str, chunk_words=180):
        toks = text.split()
        chunks = [" ".join(toks[i:i+chunk_words]) for i in range(0, len(toks), chunk_words)]
        return [c for c in chunks if len(c.split()) > 20]

    def build(self, text: str):
        self.passages = self._split(text)
        if not self.passages:
            self.index = None
            return
        embs = self.encoder.encode(self.passages, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def retrieve(self, query: str, k=6):
        if self.index is None or not self.passages:
            return []
        q = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, min(k, len(self.passages)))
        return [self.passages[i] for i in I[0] if i >= 0]

# ======================================================
#                 Factuality Verifier (NLI)
# ======================================================
class NLIVerifier:
    def __init__(self, model_name="microsoft/deberta-large-mnli"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def score(self, premise: str, hypothesis: str) -> float:
        batch = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        probs = self.model(**batch).logits.softmax(-1)[0].cpu().numpy()
        # labels: 0: contradiction, 1: neutral, 2: entailment
        return float(probs[2])

    def verify_summary(self, source_text: str, summary_text: str, thresh: float = 0.55):
        sents = [s.strip() for s in re.split(r'[.!?]+', summary_text) if s.strip()]
        results = []
        # For speed, use first 3000 chars of source; if RAG kullanılacaksa ilgili pasaj verin.
        premise = source_text[:3000]
        for s in sents:
            ent = self.score(premise, s)
            results.append({"sentence": s, "entailment": ent, "flag": ent < thresh})
        factuality = sum(r["entailment"] for r in results) / max(1, len(results))
        return {"per_sentence": results, "factuality_score": factuality}

# ======================================================
#                 Utility: Entities (optional UI)
# ======================================================
def extract_entities_rule_based(text: str) -> Dict:
    entities = {'dates': [], 'numbers': [], 'proper_nouns': []}
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b'
    ]
    for p in date_patterns:
        entities['dates'].extend(re.findall(p, text, re.IGNORECASE))
    number_patterns = [
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*%\b',
        r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?\b',
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion)\b'
    ]
    for p in number_patterns:
        entities['numbers'].extend(re.findall(p, text, re.IGNORECASE))
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities['proper_nouns'] = list(dict.fromkeys(proper_nouns))[:20]
    return entities

# ======================================================
#                 Metrics
# ======================================================
def compute_rouge(ref: str, hyp: str) -> Dict[str, float]:
    rs = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    sc = rs.score(ref, hyp)
    return {k: v.fmeasure for k, v in sc.items()}

def compute_bertscore(ref: str, hyp: str) -> float:
    P, R, F = bertscore([hyp], [ref], lang="en")
    return float(F.mean())

# ======================================================
#                 Main App
# ======================================================
def main():
    st.title("🚀 Universal Document Summarizer (Advanced)")
    st.markdown("**Models:** facebook/bart-large-cnn · DeBERTa MNLI · MiniLM RAG | **Features:** RAG, factuality, baselines, evaluation")

    # session init
    if "processor" not in st.session_state:
        st.session_state.processor = UniversalDocumentProcessor()
    if "bart" not in st.session_state:
        with st.spinner("Loading BART model..."):
            st.session_state.bart = load_bart_model()
            st.session_state.summarizer = SmartBARTSummarizer(st.session_state.bart)
    if "verifier" not in st.session_state:
        st.session_state.verifier = NLIVerifier()
    if "rag" not in st.session_state:
        st.session_state.rag = SimpleRAG()

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    method = st.sidebar.selectbox(
        "Method",
        ["BART (baseline)", "Extractive (TextRank)", "RAG + BART", "BART + Verifier", "RAG + BART + Verifier"]
    )
    summary_style = st.sidebar.selectbox("Summary Style", ['auto', 'tweet', 'short', 'medium', 'long'])
    show_entities = st.sidebar.checkbox("Show key entities (rule-based)", value=False,
                                        help="Sadece görsel destek; değerlendirme metoduna dahil değil.")
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Model:** facebook/bart-large-cnn")
    st.sidebar.write(f"**Device:** {st.session_state.bart['device']}")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Summarize", "📊 Batch", "🧪 Evaluate (with reference)", "🎯 Test Examples"])

    # ---------- Tab 1: Summarize ----------
    with tab1:
        st.header("Document Summarization")
        input_method = st.radio("Choose input method:", ["Upload File", "Paste Text", "URL"], horizontal=True)
        document_text = ""

        if input_method == "Upload File":
            up = st.file_uploader("Choose a file", type=['txt','pdf','docx','md'])
            if up is not None:
                ext = up.name.lower().split('.')[-1]
                if ext == 'pdf':
                    document_text = st.session_state.processor.read_pdf_file(up.getvalue())
                elif ext == 'docx':
                    document_text = st.session_state.processor.read_docx_file(up.getvalue())
                else:
                    document_text = st.session_state.processor.read_text_file(up.getvalue())
                if document_text:
                    c1,c2,c3 = st.columns(3)
                    with c1: st.metric("Characters", f"{len(document_text):,}")
                    with c2: st.metric("Words", f"{len(document_text.split()):,}")
                    with c3: st.metric("File Size", f"{len(up.getvalue()):,} B")
                    with st.expander("Preview"):
                        st.text_area("Document", document_text[:1000] + ("..." if len(document_text)>1000 else ""), height=200)

        elif input_method == "Paste Text":
            document_text = st.text_area("Paste text here:", height=300)

        else:  # URL
            url = st.text_input("Enter URL:", placeholder="https://example.com/article")
            if url and st.button("Extract"):
                with st.spinner("Extracting text from URL..."):
                    document_text = st.session_state.processor.read_url(url)
                if document_text:
                    st.success(f"Extracted {len(document_text.split())} words.")
                    with st.expander("Preview Extracted Text"):
                        st.text_area("Text", document_text[:1200] + ("..." if len(document_text)>1200 else ""), height=200)

        if document_text and st.button("🎯 Generate Summary", type="primary"):
            t0 = time.time()
            final_summary = ""
            used_passages = None

            # --- RAG build if needed ---
            if "RAG" in method:
                st.session_state.rag.build(document_text)
                used_passages = st.session_state.rag.retrieve(document_text[:200], k=8)
                rag_context = " ".join(used_passages) if used_passages else document_text
                final_summary = st.session_state.summarizer.summarize_bart(rag_context, style=summary_style)
            elif method == "Extractive (TextRank)":
                sent_count = max(3, int(len(document_text.split())/120))
                final_summary = textrank_summary(document_text, sentence_count=sent_count)
            else:  # plain BART
                final_summary = st.session_state.summarizer.summarize_bart(document_text, style=summary_style)

            # --- Verifier if selected ---
            ver_result = None
            if "Verifier" in method:
                premise = (" ".join(used_passages) if used_passages else document_text)[:3000]
                ver_result = st.session_state.verifier.verify_summary(premise, final_summary)
            proc_time = time.time() - t0

            # --- Display ---
            st.subheader("📝 Summary")
            st.write(final_summary)

            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Time", f"{proc_time:.2f}s")
            with c2: st.metric("Original Words", f"{len(document_text.split()):,}")
            with c3: st.metric("Summary Words", f"{len(final_summary.split()):,}")
            with c4:
                cr = len(final_summary.split())/max(1,len(document_text.split()))
                st.metric("Compression (words)", f"{cr:.1%}")

            if show_entities:
                ents = extract_entities_rule_based(document_text)
                st.subheader("🏷️ Key Entities (rule-based)")
                col1, col2 = st.columns(2)
                with col1:
                    if ents['dates']: st.write("**Dates:**", ", ".join(map(str, ents['dates'][:5])))
                    if ents['numbers']: st.write("**Numbers:**", ", ".join(map(str, ents['numbers'][:5])))
                with col2:
                    if ents['proper_nouns']: st.write("**Names:**", ", ".join(ents['proper_nouns'][:10]))

            if ver_result:
                st.subheader("🔎 Factuality Check (NLI)")
                st.metric("Avg Entailment", f"{ver_result['factuality_score']:.2f}")
                low = [r for r in ver_result["per_sentence"] if r["flag"]]
                if low:
                    st.warning("Low-confidence sentences:")
                    for r in low[:5]:
                        st.write(f"• {r['sentence']}  _(entailment={r['entailment']:.2f})_")

            # Download
            st.subheader("💾 Download")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Summary (TXT)", final_summary, file_name="summary.txt", mime="text/plain")
            with col2:
                report = f"""SUMMARY REPORT
=============
Original words: {len(document_text.split())}
Summary words : {len(final_summary.split())}
Compression   : {len(final_summary.split())/max(1,len(document_text.split())):.1%}
Method        : {method}
Time          : {proc_time:.2f}s

SUMMARY:
{final_summary}
"""
                st.download_button("Full Report", report, file_name="summary_report.txt", mime="text/plain")

    # ---------- Tab 2: Batch ----------
    with tab2:
        st.header("📊 Batch Processing")
        files = st.file_uploader("Choose multiple files", type=['txt','pdf','docx'], accept_multiple_files=True)
        if files and st.button("Process All Files"):
            rows = []
            pbar = st.progress(0)
            for i, up in enumerate(files):
                ext = up.name.lower().split('.')[-1]
                if ext == 'pdf':
                    text = st.session_state.processor.read_pdf_file(up.getvalue())
                elif ext == 'docx':
                    text = st.session_state.processor.read_docx_file(up.getvalue())
                else:
                    text = st.session_state.processor.read_text_file(up.getvalue())
                if not text:
                    continue
                t0 = time.time()
                if "RAG" in method:
                    st.session_state.rag.build(text)
                    ctx = " ".join(st.session_state.rag.retrieve(text[:200], k=8)) or text
                    summary = st.session_state.summarizer.summarize_bart(ctx, style=summary_style)
                elif method == "Extractive (TextRank)":
                    sc = max(3, int(len(text.split())/120))
                    summary = textrank_summary(text, sentence_count=sc)
                else:
                    summary = st.session_state.summarizer.summarize_bart(text, style=summary_style)
                dt = time.time() - t0
                rows.append({
                    "filename": up.name,
                    "orig_words": len(text.split()),
                    "sum_words": len(summary.split()),
                    "compression": f"{len(summary.split())/max(1,len(text.split())):.1%}",
                    "time": f"{dt:.2f}s",
                    "summary": summary
                })
                pbar.progress((i+1)/len(files))
            st.success(f"Processed {len(rows)} documents")
            df = pd.DataFrame([{k:v for k,v in r.items() if k!='summary'} for r in rows])
            st.dataframe(df, use_container_width=True)
            for r in rows:
                with st.expander(f"Summary: {r['filename']}"):
                    st.write(r["summary"])

    # ---------- Tab 3: Evaluate (with reference) ----------
    with tab3:
        st.header("🧪 Evaluate with reference summary")
        st.write("Metni ve **referans özeti** girin; ROUGE, BERTScore ve (isteğe bağlı) factuality hesaplanır.")
        src = st.text_area("Source text", height=200, key="eval_src")
        ref = st.text_area("Reference summary", height=120, key="eval_ref")
        use_verifier = st.checkbox("Also compute factuality (NLI on produced summary vs source)", value=True)
        if st.button("Run Evaluation") and src and ref:
            # Produce summary with current method
            if "RAG" in method:
                st.session_state.rag.build(src)
                ctx = " ".join(st.session_state.rag.retrieve(src[:200], k=8)) or src
                hyp = st.session_state.summarizer.summarize_bart(ctx, style=summary_style)
            elif method == "Extractive (TextRank)":
                sc = max(3, int(len(src.split())/120))
                hyp = textrank_summary(src, sentence_count=sc)
            else:
                hyp = st.session_state.summarizer.summarize_bart(src, style=summary_style)

            rdict = compute_rouge(ref, hyp)
            bsf1 = compute_bertscore(ref, hyp)
            st.subheader("Metrics")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("ROUGE-1 F1", f"{rdict['rouge1']:.3f}")
            with c2: st.metric("ROUGE-2 F1", f"{rdict['rouge2']:.3f}")
            with c3: st.metric("ROUGE-L F1", f"{rdict['rougeL']:.3f}")
            with c4: st.metric("BERTScore F1", f"{bsf1:.3f}")

            if use_verifier:
                ver = st.session_state.verifier.verify_summary(src[:3000], hyp)
                st.metric("Factuality (avg entailment)", f"{ver['factuality_score']:.2f}")
            with st.expander("Produced Summary"):
                st.write(hyp)

    # ---------- Tab 4: Examples ----------
    with tab4:
        st.header("🎯 Test Examples")

        examples = {
            "News Article": (
                "Scientists have reported a promising breakthrough that could change how Alzheimer's disease is treated. "
                "A multi-institution team led by Stanford University identified a protein, neuroprotectin-1, that appears to "
                "shield neurons from amyloid-related toxicity. In a five-year observational study of 1,842 people, higher "
                "baseline levels of the protein correlated with a significantly slower rate of cognitive decline. In mouse "
                "models, weekly injections restored memory performance on maze tasks and reduced neuroinflammation markers by "
                "38% after eight weeks. The authors caution that the mechanism is not fully understood and that long-term "
                "safety data are missing. A Phase I clinical trial to test tolerability is planned for early next year."
            ),
            "Technical Document": (
                "Deep learning systems are trained by minimizing a loss function L(θ) over large datasets using stochastic "
                "gradient descent. Convolutional neural networks (CNNs) use weight sharing and local connectivity to model "
                "spatial structure and reduce parameters, enabling translation invariance. Vision Transformers (ViT) treat "
                "images as sequences of patches and rely on self-attention to capture long-range dependencies. In NLP, "
                "encoder–decoder architectures such as BART perform denoising pretraining and are fine-tuned for tasks like "
                "summarization. Regularization techniques include dropout, label smoothing, and data augmentation, while "
                "evaluation commonly reports accuracy, F1, BLEU, or ROUGE depending on the task."
            ),
            "Business Report": (
                "The company posted strong Q2 results across all segments. Revenue rose 15% year-over-year to $2.8B, beating "
                "consensus by $200M. Cloud grew 28% to $890M as new enterprise customers onboarded ahead of schedule. "
                "Operating margin improved to 22% (vs. 19% in Q1) due to supply-chain efficiencies and a shift toward higher-"
                "margin software. Free cash flow reached $310M, up 24% YoY, and the board authorized a $250M buyback. "
                "Management raised full-year guidance to 12–14% revenue growth and EPS of $4.20–$4.40, citing a robust "
                "pipeline but warning of FX headwinds in EMEA. The outlook assumes no major macro shocks in H2."
            )
        }

        ex = st.selectbox("Choose example:", list(examples.keys()))
        if st.button("Load Example"):
            st.session_state.example_text = examples[ex]

        if hasattr(st.session_state, 'example_text'):
            st.text_area("Example Text:", st.session_state.example_text, height=220)

            if st.button("Summarize Example"):
                t0 = time.time()
                # --- summarize according to method ---
                if "RAG" in method:
                    st.session_state.rag.build(st.session_state.example_text)
                    ctx = " ".join(
                        st.session_state.rag.retrieve(st.session_state.example_text[:200], k=6)
                    ) or st.session_state.example_text
                    out = st.session_state.summarizer.summarize_bart(ctx, style=summary_style)
                elif method == "Extractive (TextRank)":
                    sc = max(3, int(len(st.session_state.example_text.split())/120))
                    out = textrank_summary(st.session_state.example_text, sentence_count=sc)
                else:
                    out = st.session_state.summarizer.summarize_bart(
                        st.session_state.example_text, style=summary_style
                    )
                dt = time.time() - t0

                # --- result and metrics---
                st.subheader("Summary")
                st.write(out)

                orig_words = len(st.session_state.example_text.split())
                sum_words  = len(out.split())
                compression = sum_words / max(1, orig_words)

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Compression (words)", f"{compression:.1%}")
                with c2: st.metric("Time", f"{dt:.2f}s")
                with c3: st.metric("Summary Words", f"{sum_words:,}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
