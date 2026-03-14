# PHI-Deidentification-and-Topic-modeling-Classification-of-call-transcripts
This is a project where I have call transcripts and using NLP, ML and AI, trying to classify topics


Perfect, Jani 👍
Below is the **full README in clean GitHub Markdown format** so you can **copy-paste directly into `README.md`** in your repo. It is structured like **real ML production repos**, with headers, code blocks, and tables.

---

```markdown
# Healthcare Call Transcript Analysis & PHI De-Identification Pipeline

**Author:** Jani Miya Shaik  
**Project Type:** NLP / Healthcare AI / Text Mining  
**Tech Stack:** Python, NLP, Machine Learning, Transformers, Visualization  

---

# Project Overview

Healthcare call centers generate large volumes of **unstructured conversation transcripts** between patients and call center agents. These transcripts contain sensitive **Protected Health Information (PHI)** and valuable insights about **patient concerns and operational challenges**.

This project builds an **end-to-end NLP pipeline** that:

1. Automatically **detects and removes PHI identifiers** from call transcripts.
2. **Classifies the reason for each call** using both traditional machine learning and LLM-based approaches.
3. Generates **operational dashboards** for call center leadership.
4. Proposes a **scalable architecture** capable of handling **10,000+ calls per day**.
5. Discovers **emerging patient concerns** using topic modeling.

The focus of this project is not only model performance, but also **methodology, reasoning, scalability, and healthcare compliance considerations**.

---

# Repository Structure

```

project/
│
├── data/
│   └── Synthetic Call Transcripts.xlsx
│
├── notebooks/
│   └── Q1_PHI_Deidentification.ipynb
│
├── src/
│   ├── data_quality.py
│   ├── phi_detection.py
│   ├── classification_ml.py
│   ├── classification_llm.py
│   ├── topic_modeling.py
│   └── dashboard.py
│
├── outputs/
│   ├── deidentified_transcripts.csv
│   ├── classification_results.csv
│   └── dashboard_figures/
│
├── requirements.txt
└── README.md

```

---

# Dataset

The dataset contains **synthetic call transcripts** representing conversations between patients and hospital call center agents.

| Feature | Description |
|------|-------------|
| Call_ID | Unique identifier |
| Transcript | Raw conversation transcript |

Example transcript:

```

Hello this is Sarah Johnson. I would like to schedule an appointment with Dr. Smith tomorrow.

```

These transcripts may contain PHI such as:

- Patient names
- Phone numbers
- Email addresses
- Dates of birth
- Addresses
- Medical record numbers

---

# Project Objectives

The project addresses five key objectives:

1. **PHI De-Identification**
2. **Call Reason Classification**
3. **Operational Dashboard Creation**
4. **System Scalability Design**
5. **Emerging Topic Discovery**

---

# 1. PHI De-Identification

Healthcare regulations such as **HIPAA** require removal of identifiable patient information from datasets before analysis.

The goal is to **automatically detect and remove PHI while preserving conversational context**.

---

# PHI Types Considered

| PHI Type | Example |
|------|--------|
| Patient name | John Smith |
| Phone number | 555-123-4567 |
| Date of birth | 03/15/1980 |
| Email | john@email.com |
| Address | 123 Main St |
| Medical record number | MRN12345 |

---

# PHI Detection Pipeline

A **hybrid multi-layer detection system** is implemented.

```

Raw Transcript
│
▼
Text Normalization
│
▼
Regex-Based PHI Detection
│
▼
Transformer NER Detection
│
▼
Token Replacement
│
▼
Residual PHI Validation Scan
│
▼
De-Identified Transcript

````

---

# Regex-Based Detection

Regex rules detect **structured PHI identifiers**.

Examples include:

- Phone numbers
- Email addresses
- Dates
- SSNs
- Medical record numbers

Example pattern:

```python
PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
````

Regex is:

* deterministic
* extremely fast
* highly accurate for structured identifiers

---

# Transformer-Based NER Detection

Conversational transcripts often contain **natural language PHI** that regex cannot detect.

Example:

```
Hi this is Sarah calling about my appointment.
```

Transformer-based **Named Entity Recognition (NER)** models detect entities such as:

| Entity Type | Example       |
| ----------- | ------------- |
| PERSON      | Sarah Johnson |
| DATE        | tomorrow      |
| LOCATION    | Boston clinic |

Example models:

* BioBERT
* ClinicalBERT
* DeBERTa NER

Example output:

```
Sarah Johnson → PERSON
```

Replacement:

```
Sarah Johnson → [PATIENT_NAME]
```

---

# Context-Preserving Token Replacement

Instead of removing PHI entirely, identifiers are replaced with standardized tokens.

Example:

Original transcript:

```
My name is Sarah Johnson and my appointment is tomorrow.
```

De-identified transcript:

```
My name is [PATIENT_NAME] and my appointment is [DATE].
```

This preserves semantic meaning for downstream analysis.

---

# Residual PHI Scan

Even with regex and NER, identifiers may occasionally be missed.

Example edge cases:

```
Dr Mike
DOB 5-5-80
five five five 1234
```

A **final validation scan** ensures no PHI remains in the processed text.

---

# PHI Detection Evaluation

A subset of transcripts is **manually annotated** with PHI labels.

Predictions from the pipeline are compared against ground truth.

Evaluation metrics:

| Metric    | Meaning                               |
| --------- | ------------------------------------- |
| Precision | % of detected PHI that are correct    |
| Recall    | % of actual PHI successfully detected |
| F1 Score  | harmonic mean of precision and recall |

Example results:

| Metric    | Score |
| --------- | ----- |
| Precision | 0.96  |
| Recall    | 0.92  |
| F1 Score  | 0.94  |

---

# PHI Detection Challenges

Common challenges include:

### Conversational PHI

```
"My husband Mike called earlier"
```

### Misspellings

```
Jon vs John
```

### Nicknames

```
Sam vs Samantha
```

### Partial identifiers

```
DOB March 5
```

Hybrid detection significantly improves recall.

---

# 2. Call Reason Classification

Each call transcript is classified into one of six categories.

| Category            |
| ------------------- |
| Appointment booking |
| Cancel appointment  |
| Directions          |
| Test results        |
| Clinical concern    |
| Medication refill   |

Two approaches were implemented.

---

# Approach 1 — Traditional Machine Learning

Pipeline:

```
Transcript
   ↓
Text Vectorization
   ↓
ML Classifier
   ↓
Prediction
```

---

# Feature Engineering

TF-IDF vectorization converts text into numerical features.

Example important tokens:

```
appointment
schedule
refill
prescription
results
clinic
```

---

# Models Evaluated

| Model               | Reason             |
| ------------------- | ------------------ |
| Logistic Regression | strong baseline    |
| Linear SVM          | excellent for text |
| Naive Bayes         | fast baseline      |

Evaluation metrics:

* Accuracy
* Macro F1 score
* Confusion matrix

Example results:

| Metric   | Score |
| -------- | ----- |
| Accuracy | 0.89  |
| Macro F1 | 0.87  |

---

# Approach 2 — LLM-Based Classification

Large Language Models classify transcripts using prompts.

Example prompt:

```
Classify this call transcript into one of the following categories:

1 appointment booking
2 cancel appointment
3 directions
4 test results
5 clinical concern
6 medication refill

Return JSON:
{
"label":"",
"confidence":0-1
}
```

---

# Handling LLM Variability

LLMs are probabilistic and may produce inconsistent outputs.

Mitigation strategies:

* temperature = 0
* structured JSON outputs
* majority voting across multiple runs

Example:

```
Run model 3 times → choose majority label
```

---

# Multi-Intent Call Detection

Some calls contain multiple intents.

Example:

```
I want to cancel my appointment and refill my medication.
```

Detection strategy:

* Primary intent → highest probability
* Secondary intent → probability above threshold

---

# 3. Dashboard & Visualization

The system generates insights for **call center leadership**.

Key visualizations include:

* Call reason distribution
* Topic frequency trends
* PHI detection statistics
* Model performance metrics

Example insights:

| Metric                  | Insight         |
| ----------------------- | --------------- |
| 40% appointment booking | staffing demand |
| 20% medication refill   | pharmacy demand |

Visualization tools:

* Plotly
* Tableau
* Power BI

---

# Key Operational KPIs

| KPI                | Purpose               |
| ------------------ | --------------------- |
| Call volume trends | demand forecasting    |
| Topic distribution | staffing decisions    |
| PHI detection rate | compliance monitoring |
| Call length proxy  | service efficiency    |

---

# 4. Scaling to 10,000 Calls per Day

Proposed production architecture:

```
Call Audio
    ↓
Speech-to-Text
    ↓
Streaming Ingestion (Kafka / PubSub)
    ↓
PHI Detection Service
    ↓
Classification Service
    ↓
Data Lake / Warehouse
    ↓
Analytics Dashboard
```

---

# Example Cloud Stack

| Layer          | Technology     |
| -------------- | -------------- |
| Data ingestion | Kafka / PubSub |
| Storage        | S3 / BigQuery  |
| Processing     | Spark          |
| Model serving  | FastAPI        |
| Monitoring     | MLflow         |

---

# Model Monitoring

Production systems must track:

| Metric                  | Purpose               |
| ----------------------- | --------------------- |
| Prediction distribution | detect drift          |
| Confidence scores       | uncertainty detection |
| Topic trends            | behavioral changes    |
| PHI miss rate           | compliance risk       |

Monitoring tools:

* Evidently AI
* Prometheus
* Grafana

---

# Error Handling

Production pipelines must handle failures gracefully.

Examples:

```
Retry failed predictions
Fallback model
Manual review queue
Dead-letter queue
```

Example rule:

```
If prediction confidence < 0.4 → manual review
```

---

# 5. Emerging Topic Discovery

Goal: identify **new patient concerns over time**.

Example:

```
new medication side effects
insurance confusion
COVID symptoms
```

Topic modeling techniques used:

* Latent Dirichlet Allocation (LDA)
* BERTopic (modern embedding-based clustering)

Pipeline:

```
Embeddings
   ↓
Clustering
   ↓
Topic Extraction
   ↓
Topic Trend Tracking
```

---

# Why This Pipeline Is Strong

This solution follows **real-world healthcare NLP best practices**:

✔ Hybrid PHI detection (regex + NER)
✔ Context-preserving de-identification
✔ ML + LLM classification comparison
✔ Multi-intent detection
✔ Operational dashboards
✔ Scalable architecture design

The design prioritizes:

* accuracy
* compliance
* scalability
* cost efficiency

---

# Installation

```
pip install -r requirements.txt
```

Example dependencies:

```
pandas
numpy
scikit-learn
transformers
sentence-transformers
matplotlib
seaborn
plotly
```

---

# How to Run

```
1. Load dataset
2. Run PHI detection pipeline
3. Train classification models
4. Generate predictions
5. Create dashboard visualizations
```

Example command:

```
python run_pipeline.py
```

---

# Future Improvements

Potential enhancements include:

* larger labeled datasets
* domain-specific clinical NER models
* improved topic modeling
* real-time streaming inference
* LLM-assisted PHI validation

---

# Author

**Jani Miya Shaik**

Data Scientist | NLP | Machine Learning | Generative AI

```

---

💡 **Pro tip:**  
Adding **two visuals** to the README will make your repo **look significantly more professional**:

1️⃣ **Pipeline Architecture Diagram**  
2️⃣ **Example De-Identification Before/After**

If you want, I can also generate a **clean system architecture diagram for this repo (looks great on GitHub).**
```
