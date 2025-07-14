# Multi‑Project Repository: NLP Pipelines & RAG‑Powered Quote Retrieval

This repository contains two end‑to‑end NLP projects:

1. **Customer Support Ticket Classification & Entity Extraction**  
2. **Semantic Quote Retrieval System with Retrieval‑Augmented Generation (RAG)**  

Each project is self‑contained with its own notebook/scripts, requirements, and instructions.

---

## 📁 Repository Structure

```

/

├── customer_support_pipeline/

│   ├── customer_support_ticket_pipeline.ipynb

│   ├── requirements.txt

│   ├── ai_dev_assignment_tickets_complex_1000.xlsx

│   └── README.md

│

└── quote_rag_system/

├── data_prep_finetune.ipynb

├── rag_pipeline_evaluate.ipynb

├── streamlit_app.py

├── requirements.txt

└── README.md

```

---

## 1. Customer Support Ticket Classification & Entity Extraction

### 📖 Project Overview
Builds a classical‑ML pipeline that:

- **Classifies** each ticket by **issue type** (multi‑class)  
- **Classifies** ticket **urgency** (Low/Medium/High)  
- **Extracts** entities: product names, dates, complaint keywords  
- Exposes an **integration** function for real‑time inference  
- (Optional) Launches a **Gradio** interface for interactive use  

### 🗒 Key Components

1. **Data Preparation**  
   - Load Excel file `ai_dev_assignment_tickets_complex_1000.xlsx`  
   - Clean text: lowercase, remove punctuation/numbers ⁠  
   - Tokenize + remove stopwords + lemmatize  
   - Handle missing values  

2. **Feature Engineering**  
   - **TF‑IDF** vectorization  
   - Ticket‑level features: text length, sentiment score  

3. **Modeling**  
   - **Random Forest** for issue‑type classification  
   - **Logistic Regression** for urgency classification  
   - Train/test split + evaluation (classification report, confusion matrix)  

4. **Entity Extraction**  
   - Rule‑based lookup from a product list  
   - Regex for dates (e.g., “15/05/2025”, “15th May”)  
   - Keyword matching for complaints (e.g., “broken”, “error”)  

5. **Integration Function**  
   ```python
   def analyze_ticket(text: str) -> dict:
       """
       Returns:
         - predicted_issue_type: str
         - predicted_urgency_level: str
         - extracted_entities: dict
       """

6. **Gradio Interface (Optional)**
   * Interactive web UI: input raw text, view predictions

### ⚙️ Setup & Usage

1. **Navigate** into the folder:
   ```bash
   cd customer_support_pipeline
   ```
2. **Create** a virtual environment & install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Launch Notebook** for development & evaluation:
   ```bash
   jupyter notebook customer_support_ticket_pipeline.ipynb
   ```
4. **Run Gradio App** (if desired):
   ```bash
   python customer_support_ticket_pipeline.ipynb  # within a notebook cell
   ```
5. **Integration** in other scripts:
   ```python
   from pipeline import analyze_ticket
   result = analyze_ticket("My device stopped working on 12th June.")
   ```

---

## 2. Semantic Quote Retrieval System with RAG

### 📖 Project Overview

Implements a Retrieval‑Augmented Generation system that:

* **Fine‑tunes** a SentenceTransformer on the Abirate/english_quotes dataset
* **Indexes** all quotes with FAISS for efficient similarity search
* **Augments** an LLM (e.g., GPT‑3.5/4) to answer natural‑language quote queries
* **Evaluates** retrieval quality via sample queries & (optionally) RAGAS/Quotient
* **Deploys** a **Streamlit** web app for end‑user interaction

### 🗒 Key Components

1. **Data Preparation**

   ```python
   from datasets import load_dataset
   ds = load_dataset("Abirate/english_quotes")
   df = pd.DataFrame(ds["train"])
   df["quote_clean"] = df.quote.str.strip().dropna()
   ```
2. **Model Fine‑Tuning**

   * Base model: `all-MiniLM-L6-v2`
   * Loss: `MultipleNegativesRankingLoss` on (quote, author+tags) pairs
   * Output: `fine_tuned_quote_sbert/`
3. **Indexing with FAISS**

   ```python
   embeddings = model.encode(df.quote_clean.tolist())
   index = faiss.IndexFlatL2(embeddings.shape[1])
   index.add(embeddings)
   ```
4. **RAG Pipeline**

   * **Retriever** : FAISS-wrapped LangChain retriever
   * **Generator** : OpenAI LLM via `langchain.chains.RetrievalQA`
   * **Prompt** : instruct LLM to output JSON {quotes, authors, tags, similarity_scores}
5. **Evaluation**

   * Sample queries (“Einstein insanity quotes”, “accomplishment”)
   * Print JSON responses + source documents
   * (Optional) integrate with RAGAS, Quotient, or Arize for quantitative metrics
6. **Streamlit App** (`streamlit_app.py`)

   ```bash
   streamlit run streamlit_app.py
   ```

   * Query input box
   * JSON response display
   * Top‑k quotes with score captions

### ⚙️ Setup & Usage

1. **Navigate** into the folder:
   ```bash
   cd quote_rag_system
   ```
2. **Create** virtual environment & install:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Data Prep & Fine‑Tune**
   ```bash
   jupyter notebook data_prep_finetune.ipynb
   ```
4. **Build & Evaluate RAG**
   ```bash
   jupyter notebook rag_pipeline_evaluate.ipynb
   ```
5. **Launch Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🎥 Demo Video

A short screencast demonstrating:

* Data loading & cleaning
* Model training & indexing
* Example RAG queries & evaluation
* Interactive Streamlit application

---

## 📝 Notes & Best Practices

* **Reproducibility:** use fixed random seeds for dataset splits & model training.
* **Performance:** adjust `max_features` for TF‑IDF or `k` in retrieval to trade off accuracy vs. speed.
* **Extensibility:** swap in more powerful LLMs (e.g., GPT‑4, Llama 3) or vector stores (Chroma, Pinecone).
* **Security:** do not commit API keys; load them via environment variables (`OPENAI_API_KEY`).

---

Thank you for exploring these pipelines!

For questions or issues, please open a GitHub issue or contact the maintainer.
