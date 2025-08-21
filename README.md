# Agentic_RAG_with_Websearch_feature

**Gemini RAG Agent** is a Streamlit-based AI assistant that allows users to upload PDF documents, query their content, and get answers from both the uploaded PDFs and the web. It combines **retrieval-augmented generation (RAG)** with real-time web search for accurate and informative responses.

---

## Features

- **PDF Document Analysis**
  - Upload multiple PDFs at once.
  - Automatically split PDFs into chunks for AI search.
  - Search content across all uploaded documents.
  - Highlights relevant terms and shows page numbers.

- **Web Search Integration**
  - Uses Google Serper API for real-time web search.
  - Returns summarized and formatted results.
  - Only used when PDF content is insufficient or outdated.

- **AI Chat Interface**
  - Powered by Google Gemini 2.0.
  - Retains conversation context for follow-up questions.
  - Summarizes content from PDFs or chat history.
  - Shows sources (documents or web URLs) alongside answers.
  - Expandable view for intermediate reasoning steps.

- **Document & Chat Management**
  - Reset all uploaded PDFs.
  - Clear chat history.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Streamlit
- Environment variables:
  - `GOOGLE_API_KEY` – for Google Gemini LLM access.
  - `SERPER_API_KEY` – for web search.

### Installation

```bash
git clone <repository_url>
cd gemini-rag-agent
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

- Upload PDF documents via the sidebar.
- Type questions in the chat box.
- View answers with sources and optional reasoning steps.

---

## Usage Example

1. Upload PDFs like `Report1.pdf` and `ResearchPaper.pdf`.
2. Ask a question in the chat:

```
What are the main points about climate change in these documents?
```

3. The AI will:
   - Search the uploaded PDFs first.
   - Use web search if necessary.
   - Highlight key terms in the content.
   - Provide sources with document names, page numbers, or URLs.

4. Ask for a summary:

```
Summarize the main findings about climate change.
```

---

## Project Structure

```
gemini-rag-agent/
│
├─ app.py                # Main Streamlit app
├─ requirements.txt      # Python dependencies
├─ .env                  # API keys
├─ chroma_store/         # Persisted vectorstore for PDFs
└─ README.md             # Project documentation
```

---

## Technologies Used

- **Streamlit** – Web app framework.
- **LangChain** – Orchestrates LLM, tools, and memory.
- **Google Gemini 2.0** – LLM for natural language responses.
- **HuggingFace Embeddings** – Converts text into embeddings.
- **Chroma Vectorstore** – Stores and retrieves document embeddings.
- **Google Serper API** – Web search for current information.

---

## License

MIT License – See LICENSE file for details.
