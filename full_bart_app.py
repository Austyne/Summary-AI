"""
Full BART Document Summarizer - Streamlit App
============================================
Complete system with full facebook/bart-large-cnn model
"""

import torch
import pandas as pd
import numpy as np
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from rouge_score import rouge_scorer
from datasets import load_dataset

# Document processing
import PyPDF2
try:
    from docx import Document
except ImportError:
    Document = None
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Visualization and UI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Streamlit configuration
st.set_page_config(
    page_title="Universal BART Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UniversalDocumentProcessor:
    """Handles all types of document input processing"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.md']
        
    def read_text_file(self, file_content) -> str:
        """Read text from uploaded file"""
        try:
            return str(file_content, 'utf-8')
        except UnicodeDecodeError:
            return str(file_content, 'latin-1')
    
    def read_pdf_file(self, file_content) -> str:
        """Extract text from PDF file content"""
        try:
            import io
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def read_docx_file(self, file_content) -> str:
        """Extract text from DOCX file content"""
        try:
            if Document is None:
                st.error("python-docx not installed. Install with: pip install python-docx")
                return ""
            
            import io
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    def read_url(self, url: str) -> str:
        """Extract text from web pages"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            st.error(f"Error reading URL: {e}")
            return ""


@st.cache_resource
def load_bart_model():
    """Load Full BART model with Streamlit caching"""
    model_name = "facebook/bart-large-cnn"
    print(f"Loading Full BART model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Create pipeline
    summarizer_pipeline = pipeline(
        "summarization", 
        model=model, 
        tokenizer=tokenizer,
        device=0 if device == 'cuda' else -1
    )
    
    print(f"Full BART model loaded successfully on {device}")
    return {
        'pipeline': summarizer_pipeline,
        'tokenizer': tokenizer,
        'model': model,
        'device': device
    }


class SmartBARTSummarizer:
    """Advanced BART-based summarizer with multiple modes and smart features"""
    
    def __init__(self, bart_components):
        self.pipeline = bart_components['pipeline']
        self.tokenizer = bart_components['tokenizer']
        self.model = bart_components['model']
        self.device = bart_components['device']
        
        # Summary presets
        self.summary_presets = {
            'tweet': {'max_length': 50, 'min_length': 20, 'description': 'Twitter-style brief summary'},
            'short': {'max_length': 100, 'min_length': 40, 'description': 'Short paragraph summary'},
            'medium': {'max_length': 200, 'min_length': 80, 'description': 'Standard summary'},
            'long': {'max_length': 400, 'min_length': 150, 'description': 'Detailed summary'},
            'auto': {'max_length': None, 'min_length': None, 'description': 'Automatically adjusted length'}
        }
    
    def _smart_length_calculation(self, text: str) -> Dict[str, int]:
        """Calculate optimal summary length based on input text characteristics"""
        word_count = len(text.split())
        
        # Smart length calculation
        if word_count < 100:
            ratio = 0.7
        elif word_count < 500:
            ratio = 0.4
        elif word_count < 1500:
            ratio = 0.25
        elif word_count < 5000:
            ratio = 0.15
        else:
            ratio = 0.1
        
        target_words = max(30, int(word_count * ratio))
        max_length = min(400, int(target_words * 1.3))
        min_length = max(20, int(target_words * 0.7))
        
        return {
            'max_length': max_length,
            'min_length': min_length,
            'target_words': target_words,
            'compression_ratio': ratio
        }
    
    def _chunk_long_text(self, text: str, max_tokens: int = 900) -> List[str]:
        """Split long text into chunks that BART can handle"""
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= max_chars:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize(self, 
                 text: str, 
                 style: str = 'auto',
                 custom_length: Optional[Dict] = None,
                 extract_entities: bool = True) -> Dict:
        """Generate summary with multiple options"""
        start_time = time.time()
        
        # Prepare length parameters
        if style == 'custom' and custom_length:
            length_params = custom_length
        elif style == 'auto':
            length_params = self._smart_length_calculation(text)
        else:
            length_params = self.summary_presets[style].copy()
        
        # Handle long documents by chunking
        chunks = self._chunk_long_text(text)
        chunk_summaries = []
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:
                    continue
                
                status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
                progress_bar.progress((i + 1) / len(chunks))
                
                chunk_max_length = length_params['max_length']
                chunk_min_length = length_params['min_length']
                
                if len(chunks) > 1:
                    chunk_max_length = min(200, chunk_max_length)
                    chunk_min_length = min(chunk_min_length, chunk_max_length - 20)
                
                result = self.pipeline(
                    chunk,
                    max_length=chunk_max_length,
                    min_length=chunk_min_length,
                    do_sample=False,
                    truncation=True
                )
                
                chunk_summaries.append(result[0]['summary_text'])
            
            status_text.text("Finalizing summary...")
            
            # Combine chunk summaries
            if len(chunk_summaries) > 1:
                combined_text = " ".join(chunk_summaries)
                
                if len(combined_text.split()) > length_params.get('max_length', 200):
                    final_result = self.pipeline(
                        combined_text,
                        max_length=length_params['max_length'],
                        min_length=length_params['min_length'],
                        do_sample=False,
                        truncation=True
                    )
                    final_summary = final_result[0]['summary_text']
                else:
                    final_summary = combined_text
            else:
                final_summary = chunk_summaries[0] if chunk_summaries else "Unable to generate summary."
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Extract key entities if requested
            entities = self._extract_entities(text) if extract_entities else {}
            
            # Calculate metrics
            metrics = self._calculate_metrics(text, final_summary)
            
            result = {
                'summary': final_summary,
                'original_length': len(text),
                'summary_length': len(final_summary),
                'compression_ratio': len(final_summary) / len(text) if len(text) > 0 else 0,
                'processing_time': time.time() - start_time,
                'style_used': style,
                'chunks_processed': len(chunks),
                'entities': entities,
                'metrics': metrics,
                'length_params': length_params
            }
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            result = {
                'summary': f"Error generating summary: {str(e)}",
                'error': str(e),
                'processing_time': time.time() - start_time,
                'original_length': len(text),
                'summary_length': 0,
                'compression_ratio': 0
            }
        
        return result
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract key entities using rule-based approach"""
        entities = {
            'dates': [],
            'numbers': [],
            'proper_nouns': [],
            'organizations': []
        }
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b'
        ]
        
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Numbers and percentages
        number_patterns = [
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*%\b',
            r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?\b',
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion)\b'
        ]
        
        for pattern in number_patterns:
            entities['numbers'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities['proper_nouns'] = list(set(proper_nouns[:20]))
        
        return entities
    
    def _calculate_metrics(self, original: str, summary: str) -> Dict:
        """Calculate quality metrics"""
        return {
            'word_count_original': len(original.split()),
            'word_count_summary': len(summary.split()),
            'sentence_count_original': len(re.split(r'[.!?]+', original)),
            'sentence_count_summary': len(re.split(r'[.!?]+', summary)),
            'avg_sentence_length': len(summary.split()) / max(1, len(re.split(r'[.!?]+', summary))),
            'lexical_diversity': len(set(summary.lower().split())) / max(1, len(summary.split()))
        }


# Main Streamlit App
def main():
    st.title("🚀 Universal BART Document Summarizer")
    st.markdown("*AI-powered summarization using facebook/bart-large-cnn*")
    
    # Initialize session state
    if 'bart_model' not in st.session_state:
        with st.spinner("🤖 Loading Full BART Model... (this will take 2-3 minutes first time)"):
            st.session_state.bart_model = load_bart_model()
            st.session_state.summarizer = SmartBARTSummarizer(st.session_state.bart_model)
            st.session_state.processor = UniversalDocumentProcessor()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    summary_style = st.sidebar.selectbox(
        "Summary Style",
        ['auto', 'tweet', 'short', 'medium', 'long'],
        help="Choose the length and style of summary"
    )
    
    extract_entities = st.sidebar.checkbox("Extract Key Entities", value=True)
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 Model Info**")
    st.sidebar.write(f"Model: facebook/bart-large-cnn")
    st.sidebar.write(f"Device: {st.session_state.bart_model['device']}")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📄 Summarize", "📊 Batch Process", "🎯 Test Examples"])
    
    with tab1:
        st.header("Document Summarization")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text", "URL"]
        )
        
        document_text = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'pdf', 'docx', 'md'],
                help="Supported formats: TXT, PDF, DOCX, MD"
            )
            
            if uploaded_file is not None:
                # Process document based on file type
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if file_extension == 'pdf':
                    document_text = st.session_state.processor.read_pdf_file(uploaded_file.getvalue())
                elif file_extension == 'docx':
                    document_text = st.session_state.processor.read_docx_file(uploaded_file.getvalue())
                else:  # txt, md
                    document_text = st.session_state.processor.read_text_file(uploaded_file.getvalue())
                
                if document_text:
                    # Show document info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Characters", f"{len(document_text):,}")
                    with col2:
                        st.metric("Words", f"{len(document_text.split()):,}")
                    with col3:
                        st.metric("File Size", f"{len(uploaded_file.getvalue()):,} bytes")
                    
                    # Show preview
                    with st.expander("Preview Document"):
                        st.text_area("Document Preview", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)
        
        elif input_method == "Paste Text":
            document_text = st.text_area(
                "Paste your text here:",
                height=300,
                help="Copy and paste any text you want to summarize"
            )
        
        elif input_method == "URL":
            url = st.text_input(
                "Enter URL:",
                placeholder="https://example.com/article"
            )
            
            if url and st.button("Extract Text from URL"):
                with st.spinner("Extracting text from URL..."):
                    document_text = st.session_state.processor.read_url(url)
                    
                if document_text:
                    st.success(f"Extracted {len(document_text.split())} words from URL")
                    with st.expander("Preview Extracted Text"):
                        st.text_area("Extracted Text", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)
                else:
                    st.error("Failed to extract text from URL")
        
        # Generate summary
        if document_text and st.button("🎯 Generate Summary", type="primary"):
            summary_result = st.session_state.summarizer.summarize(
                document_text,
                style=summary_style,
                extract_entities=extract_entities
            )
            
            # Display results
            st.subheader("📝 Summary")
            st.write(summary_result['summary'])
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Compression Ratio", f"{summary_result['compression_ratio']:.1%}")
            with col2:
                st.metric("Processing Time", f"{summary_result['processing_time']:.2f}s")
            with col3:
                st.metric("Original Words", f"{len(document_text.split()):,}")
            with col4:
                st.metric("Summary Words", f"{len(summary_result['summary'].split()):,}")
            
            # Entities
            if extract_entities and summary_result.get('entities'):
                st.subheader("🏷️ Key Entities")
                entities = summary_result['entities']
                
                col1, col2 = st.columns(2)
                with col1:
                    if entities['dates']:
                        st.write("**Dates:**", ", ".join(entities['dates'][:5]))
                    if entities['numbers']:
                        st.write("**Numbers:**", ", ".join(entities['numbers'][:5]))
                with col2:
                    if entities['proper_nouns']:
                        st.write("**Names:**", ", ".join(entities['proper_nouns'][:10]))
            
            # Download options
            st.subheader("💾 Download")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Summary (TXT)",
                    summary_result['summary'],
                    file_name="summary.txt",
                    mime="text/plain"
                )
            with col2:
                report = f"""SUMMARY REPORT
=============
Original Length: {summary_result['original_length']} characters
Summary Length: {summary_result['summary_length']} characters
Compression Ratio: {summary_result['compression_ratio']:.1%}
Processing Time: {summary_result['processing_time']:.2f}s
Style Used: {summary_result['style_used']}

SUMMARY:
{summary_result['summary']}
"""
                st.download_button(
                    "Download Full Report",
                    report,
                    file_name="summary_report.txt",
                    mime="text/plain"
                )
    
    with tab2:
        st.header("📊 Batch Processing")
        st.write("Upload multiple files for batch summarization")
        
        uploaded_files = st.file_uploader(
            "Choose multiple files",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process All Files"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Extract text based on file type
                file_extension = uploaded_file.name.lower().split('.')[-1]
                
                if file_extension == 'pdf':
                    text = st.session_state.processor.read_pdf_file(uploaded_file.getvalue())
                elif file_extension == 'docx':
                    text = st.session_state.processor.read_docx_file(uploaded_file.getvalue())
                else:
                    text = st.session_state.processor.read_text_file(uploaded_file.getvalue())
                
                if text:
                    # Generate summary
                    result = st.session_state.summarizer.summarize(text, summary_style, extract_entities=False)
                    results.append({
                        'filename': uploaded_file.name,
                        'original_words': len(text.split()),
                        'summary_words': len(result['summary'].split()),
                        'compression': f"{result['compression_ratio']:.1%}",
                        'processing_time': f"{result['processing_time']:.2f}s",
                        'summary': result['summary']
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Complete!")
            st.success(f"Processed {len(results)} documents")
            
            # Display results table
            df = pd.DataFrame([{k: v for k, v in r.items() if k != 'summary'} for r in results])
            st.dataframe(df, use_container_width=True)
            
            # Show individual summaries
            for result in results:
                with st.expander(f"Summary: {result['filename']}"):
                    st.write(result['summary'])
    
    with tab3:
        st.header("🎯 Test Examples")
        st.write("Try the summarizer with pre-loaded examples")
        
        examples = {
            "News Article": """
            Scientists have made a groundbreaking discovery that could revolutionize the treatment of Alzheimer's disease. 
            Researchers at Stanford University have identified a new protein that appears to protect brain cells from the 
            damage associated with this neurodegenerative condition. The study, published in Nature Medicine, shows that 
            patients with higher levels of this protein, called neuroprotectin-1, show significantly slower cognitive 
            decline over a five-year period. The research team tested the protein in laboratory mice and found that it 
            could reverse memory loss and reduce brain inflammation. Clinical trials in humans are expected to begin 
            within the next two years, offering hope to millions of families affected by Alzheimer's disease.
            """,
            "Technical Document": """
            Machine learning algorithms have become increasingly sophisticated in recent years, with deep learning 
            representing a significant advancement in artificial intelligence. Neural networks, inspired by the structure 
            of the human brain, consist of interconnected nodes that process information in layers. These systems excel 
            at pattern recognition tasks, such as image classification and natural language processing. The training 
            process involves feeding large datasets to the network, allowing it to learn complex relationships between 
            inputs and outputs. Convolutional neural networks are particularly effective for image analysis, while 
            recurrent neural networks are well-suited for sequential data like text and speech.
            """,
            "Business Report": """
            The quarterly earnings report shows strong performance across all major business segments. Revenue increased 
            by 15% year-over-year to $2.8 billion, exceeding analyst expectations of $2.6 billion. The cloud computing 
            division was the strongest performer, growing 28% and contributing $890 million to total revenue. Operating 
            margins improved to 22%, up from 19% in the previous quarter, due to operational efficiencies and cost 
            reduction initiatives. The company raised its full-year guidance, now expecting revenue growth of 12-14% 
            and earnings per share of $4.20-$4.40. Management remains optimistic about the outlook for the remainder 
            of the fiscal year.
            """
        }
        
        selected_example = st.selectbox("Choose an example:", list(examples.keys()))
        
        if st.button("Load Example"):
            st.session_state.example_text = examples[selected_example]
        
        if hasattr(st.session_state, 'example_text'):
            st.text_area("Example Text:", st.session_state.example_text, height=200)
            
            if st.button("Summarize Example"):
                result = st.session_state.summarizer.summarize(
                    st.session_state.example_text,
                    style=summary_style,
                    extract_entities=extract_entities
                )
                
                st.subheader("Summary:")
                st.write(result['summary'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Compression", f"{result['compression_ratio']:.1%}")
                with col2:
                    st.metric("Time", f"{result['processing_time']:.2f}s")
                with col3:
                    st.metric("Words", f"{len(result['summary'].split())}")


if __name__ == "__main__":
    main()