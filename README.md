# 📝 Summary-AI

**An Advanced AI-Powered Document Summarization System**

Summary-AI is a sophisticated text summarization application built with the state-of-the-art BART (Bidirectional and Auto-Regressive Transformers) model. This powerful tool can intelligently summarize documents, web pages, and text content with multiple customization options and advanced features.

## 🚀 Features

### 📄 Multi-Format Document Support
- **PDF Documents**: Extract and summarize PDF files
- **Word Documents**: Process DOCX files seamlessly  
- **Text Files**: Handle TXT and Markdown files
- **Web Content**: Extract and summarize content from URLs
- **Direct Text Input**: Paste text directly for quick summarization

### 🎯 Smart Summarization Modes
- **Tweet Style** (20-50 words): Ultra-brief summaries perfect for social media
- **Short Summary** (40-100 words): Concise paragraph summaries
- **Medium Summary** (80-200 words): Standard detailed summaries
- **Long Summary** (150-400 words): Comprehensive detailed summaries
- **Auto Mode**: Intelligent length adjustment based on document characteristics
- **Custom Length**: User-defined summary parameters

### 🧠 Advanced AI Features
- **Full BART-Large-CNN Model**: State-of-the-art Facebook transformer model
- **Smart Chunking**: Handles long documents by intelligent text segmentation
- **Entity Extraction**: Identifies key dates, numbers, and proper nouns
- **Compression Analytics**: Detailed metrics on summarization efficiency
- **GPU Acceleration**: Automatic GPU utilization for faster processing

### 📊 Batch Processing
- Process multiple documents simultaneously
- Comprehensive results dashboard
- Export batch results to various formats
- Progress tracking and status updates

### 💡 User Experience
- **Streamlit Web Interface**: Clean, intuitive web-based UI
- **Real-time Processing**: Live progress indicators and status updates
- **Interactive Preview**: Document preview before summarization
- **Download Options**: Export summaries and detailed reports
- **Pre-loaded Examples**: Test with sample documents

## 🛠️ Technology Stack

- **AI/ML**: Transformers (BART), PyTorch, Hugging Face
- **Web Framework**: Streamlit
- **Document Processing**: PyPDF2, python-docx, BeautifulSoup
- **Data Science**: NumPy, Pandas, NLTK, spaCy
- **Visualization**: Plotly
- **Web Scraping**: Requests, BeautifulSoup

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- At least 4GB RAM
- Internet connection (for model download)

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Austyne/Summary-AI.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy models:**
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```

### Running the Application

1. **Start the Streamlit server:**
   ```bash
   streamlit run full_bart_app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Start summarizing!** Upload a document, paste text, or enter a URL

## 📖 Usage Guide

### Single Document Summarization
1. Choose your input method (Upload File/Paste Text/URL)
2. Select summarization style and options
3. Click "Generate Summary"
4. Review results and download if needed

### Batch Processing
1. Navigate to the "Batch Processing" tab
2. Upload multiple files
3. Click "Process All Files"
4. View results table and individual summaries

### Testing Examples
1. Go to the "Test Examples" tab
2. Select from pre-loaded sample texts
3. Experiment with different summarization modes

## 🎛️ Configuration Options

### Summary Styles
- **Tweet**: Perfect for social media (20-50 words)
- **Short**: Quick overviews (40-100 words)  
- **Medium**: Standard summaries (80-200 words)
- **Long**: Detailed summaries (150-400 words)
- **Auto**: Smart length based on content
- **Custom**: User-defined parameters

### Advanced Features
- **Entity Extraction**: Toggle to identify key information
- **Compression Metrics**: View detailed analytics
- **Processing Time**: Monitor performance
- **GPU Utilization**: Automatic hardware optimization

## 📊 Performance Metrics

The application provides comprehensive analytics:
- **Compression Ratio**: How much the text was reduced
- **Processing Time**: Time taken for summarization
- **Word Count Comparison**: Original vs. summary length
- **Entity Recognition**: Key dates, numbers, and names extracted

## 🔧 Technical Architecture

### Core Components
- **UniversalDocumentProcessor**: Handles all document types and URL extraction
- **SmartBARTSummarizer**: Advanced BART-based summarization engine
- **Streamlit Interface**: User-friendly web application

### Model Details
- **Base Model**: facebook/bart-large-cnn
- **Architecture**: Encoder-decoder transformer
- **Training**: Pre-trained on CNN/DailyMail dataset
- **Capabilities**: Abstractive and extractive summarization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and BART model
- **Facebook Research** for the original BART paper and model
- **Streamlit** for the excellent web framework
- **PyTorch** for the deep learning infrastructure

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation
- Review the example usage in the app

## 🔄 Version History

- **v1.0.0**: Initial release with full BART integration
- Advanced document processing capabilities
- Multi-format support and batch processing
- Comprehensive web interface

---

**Made with ❤️ and 🤖 AI lol**

*Transform your documents into concise, intelligent summaries with the power of advanced AI.*
