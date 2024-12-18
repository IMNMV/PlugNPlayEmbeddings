# PlugNPlayEmbeddings

Text Analysis with Embeddings - This application provides semantic search and clustering analysis for text documents using Nomic embeddings and Ollama.

## Prerequisites
* Python 3.9+
* Ollama installed (https://ollama.com)
* Nomic-embed-text model pulled in Ollama: https://ollama.com/library/nomic-embed-text

## Setup Instructions

1. Install Ollama:
   * Visit https://ollama.com
   * Follow installation instructions for your OS

2. Pull the Nomic embeddings model using the command line interface:
   ```bash
   ollama pull nomic-embed-text

   # Confirm it worked correctly - should see a lot of numbers (embeddings) populate 
   curl http://localhost:11434/api/embeddings -d '{
    "model": "nomic-embed-text",
    "prompt": "test"
    }'
   ```
  

3. Create and activate virtual environment:
   
    For macOS/Linux:
      ```bash
      # Create virtual environment
      python -m venv plugnplay_env
         
      # Activate virtual environment
      source plugnplay_env/bin/activate
      ```
     
    For Windows:
      ```bash
      # Create virtual environment
      python -m venv plugnplay_env
        
      # Activate virtual environment
      .\plugnplay_env\Scripts\activate
      ```

4. Install Dependencies and run:
   ```bash
   # Install requirements
   pip install -r requirements.txt

   # Run the application
   streamlit run app.py
   ```

5. Deactivate when done
   ```bash
   deactivate
   ```


# Simple Text Analysis Tool 🔍

This tool helps you understand and compare text documents in a super simple way. Think of it like having a smart friend who reads your documents and tells you what they're about!

## Features 🌟

### 1. Document Analysis
- **Chunk Breaking**: Splits big documents into smaller, bite-sized pieces (like cutting a sandwich into smaller pieces)
- **Smart Search**: Find specific information in your documents by asking questions in normal language
- **Topic Discovery**: Automatically finds the main themes in your documents (like finding out what a book is about without reading it)
- **Pretty Pictures**: Shows cool visualizations to help you understand your documents better

### 2. Document Comparison
- **Side-by-Side Look**: Compare two documents to see how similar or different they are
- **Topic Sharing**: See what themes two documents share (like finding out what two stories have in common)
- **Similarity Maps**: Shows you which parts of the documents are alike using colors and pictures

## How to Use It 🎮

### Single Document Analysis
1. Click "Single Document Analysis"
2. Upload your document (it can be .txt, .md, .csv, or .json)
3. Choose what you want to do:
   - **Search**: Type a question to find specific parts of your document
   - **Find Topics**: Click "Extract Topics" to see what your document is about
   - **See Pictures**: Look at the cool maps and charts that show how your document is organized

### Comparing Two Documents
1. Click "Document Comparison"
2. Upload two documents you want to compare
3. Click "Compare Documents" to see:
   - How similar the documents are
   - What topics they share
   - Cool visualizations showing their relationships

## Understanding the Results 🧐

### Search Results
- **Similarity Score**: The higher the number (closer to 1.0), the more relevant that part is to your question
- **Chunks**: These are the pieces of text that match what you're looking for

### Topic Analysis
- Each topic shows you:
  - The main ideas found in your document
  - Which parts of the document talk about each topic
  - How strong each topic is in different parts

### Visualizations
1. **Network View**: 
   - Each dot is a piece of your document
   - Lines between dots mean those pieces are similar
   - Closer dots = more similar content

2. **Heatmap**:
   - Blue colors show how similar different parts are
   - Darker blue = more similar
   - Lighter blue = less similar

3. **TSNE Map**:
   - Shows how different parts of your document relate to each other
   - Closer points = more similar content
   - Different colors = different documents

### Tips for Best Results 🎯
- Use documents that are readable text files
- For best topic analysis, use documents with clear themes
- When searching, try to ask specific questions
- Play with the settings (like chunk size) to get better results

## Settings You Can Change ⚙️
- **Chunk Size**: How big each piece of text should be (bigger isn't always better!)
- **Number of Topics**: How many main themes to look for
- **Search Threshold**: How picky you want the search to be
- **Network Threshold**: How strong connections need to be to show up in the network view

That's it! Now you can explore your documents like a pro! 🎉
