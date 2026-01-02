# Enterprise-Grade RAG System Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for intelligent document question-answering, particularly designed for financial regulatory documents. Here's the complete flow:

## Step-by-Step Process

### 1. **Document Ingestion** (`POST /api/v1/ingest`)

**Input:**
- PDF or text file (e.g., financial reports, regulatory filings)
- Metadata JSON (document_id, source_uri, filing_type, fiscal_period, etc.)

**Processing:**
1. **Table Extraction** (`TableExtractionService`)
   - Uses Docling library to parse document
   - Extracts tables (e.g., income statements) as CSV format
   - Extracts clean text content

2. **Hierarchical Chunking** (`HierarchicalChunker`)
   - Splits document into multi-level chunks:
     - **Title**: Document heading (top-level)
     - **Section**: Major sections
     - **Paragraph**: Individual paragraphs within sections
     - **Table**: Extracted financial tables
   - Creates parent-child relationships for context

3. **Embedding Generation** (`EmbeddingService`)
   - Generates dense vectors using `sentence-transformers/all-mpnet-base-v2`
   - Creates sparse tokens for BM25 (keyword-based search)
   - Each chunk gets a 768-dimensional embedding vector

4. **Storage** (`PGVectorStore`)
   - Stores chunks in PostgreSQL with pgvector extension
   - Indexes: dense embeddings, sparse tokens, metadata
   - Enables hybrid search capabilities

**Output:**
```json
{
  "document_id": "uuid",
  "chunks_indexed": 150,
  "tables_indexed": 5,
  "elapsed_ms": 2500
}
```

---

### 2. **Query Answering** (`POST /api/v1/query`)

**Input:**
```json
{
  "question": "What was the Q4 revenue guidance?",
  "top_k": 8,
  "audit_trail": true
}
```

**Processing:**

1. **Hybrid Retrieval** (`HybridRetriever`)
   
   **a. Dense Search:**
   - Embeds user question into 768-dim vector
   - Performs cosine similarity search in pgvector
   - Returns top semantically similar chunks
   
   **b. Sparse Search (BM25):**
   - Tokenizes question
   - Calculates BM25 scores: $\text{BM25}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$
   - Returns keyword-matched chunks
   
   **c. Reciprocal Rank Fusion (RRF)** (`app/utils/rrf.py`)
   - Fuses dense + sparse results: $\text{score}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + r(d)}$
   - Default k=60 (dampening factor)
   - Produces unified ranked list

2. **Answer Generation** (`QueryOrchestrator`)
   
   **With LLM (if `OPENAI_API_KEY` provided):**
   - Formats top-k chunks as context
   - Uses LangChain with ChatGPT-4o-mini
   - Prompt: "Answer using only provided context and cite chunk_ids"
   - Returns grounded answer with citations like `[chunk-uuid-123]`
   
   **Without LLM (fallback):**
   - Returns extractive answer from best-matching chunk
   - Includes disclaimer about configuration limits

**Output:**
```json
{
  "answer": "Q4 revenue guidance is $12.5B-$13B [chunk-uuid-789]",
  "references": [
    {
      "chunk_id": "chunk-uuid-789",
      "document_id": "doc-456",
      "content": "Revenue guidance for Q4 2024 is $12.5-$13 billion.",
      "score": 0.89,
      "level": "paragraph",
      "metadata": {...}
    }
  ],
  "grounded": true,
  "latency_ms": 450
}
```

---

### 3. **Evaluation** (`POST /api/v1/evaluate`)

**Input:**
```json
[
  {
    "question": "What was the Q4 revenue guidance?",
    "answer": "Guidance is $12.5B-$13B.",
    "ground_truth": "Revenue guidance for Q4 2024 is $12.5-$13 billion.",
    "citations": ["Revenue guidance for Q4 2024 is $12.5-$13 billion."]
  }
]
```

**Processing** (`EvaluationService`):

1. **RAGAS Metrics** (requires ragas library):
   - **Faithfulness**: Does answer match citations?
   - **Answer Relevancy**: Is answer relevant to question?

2. **Custom Metrics**:
   - **Groundedness**: % of answers where citations contain ground_truth
   - **Recall@K**: % of queries where ground_truth appears in top-K results

**Output:**
```json
{
  "samples_evaluated": 50,
  "metrics": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "groundedness": 0.85,
    "recall_at_k": 0.90
  },
  "generated_at": "2024-01-15T10:30:00Z"
}
```

---

## Key Technologies

- **FastAPI**: REST API framework (`app/main.py`, `app/api/routers.py`)
- **PostgreSQL + pgvector**: Vector database (`app/services/pgvector_store.py`)
- **LangChain**: LLM orchestration (`app/services/orchestration.py`)
- **Sentence Transformers**: Embeddings (`app/services/embedding.py`)
- **Docling**: Document parsing (`app/services/table_extractor.py`)
- **BM25**: Sparse retrieval (`app/services/hybrid_retriever.py`)

## Architecture Flow

```
Document → Docling → Chunks → Embeddings → PostgreSQL
                ↓                              ↓
Question → Embed → Dense Search ─┐            │
           ↓                      │            │
      Tokenize → BM25 Search ────┴→ RRF Fusion│
                                      ↓        │
                                  Top-K ←──────┘
                                      ↓
                                  LangChain
                                      ↓
                                  Answer + Citations
```

The system ensures **grounded answers** by requiring LLM responses to cite specific chunk IDs, making all claims traceable back to source documents.
