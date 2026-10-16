# Smart University Assistant
### AI-Powered Academic Advising & Arabic Course Recommendation System

**Graduation Project | Artificial Intelligence & Robotics | Al-Balqa Applied University**

An intelligent academic advising system combining **Hybrid Retrieval-Augmented Generation (RAG)**, **Natural Language Processing (NLP)**, **Semantic Search**, and **Rule-Based Decision Making** to support university students in academic planning and course registration.

**Core Technologies:** Python · Flask · FAISS · Sentence Transformers · Multilingual E5 · Qwen2.5-7B · NLP

---

## 1. Project Overview

Smart University Assistant is an AI-powered academic advising platform developed to help university students understand academic requirements, retrieve course information, and make informed semester registration decisions.

Traditional academic advising often requires students to manually review course descriptions, prerequisites, credit-hour requirements, and university regulations. This process can be time-consuming and difficult, particularly when students need to evaluate multiple registration constraints.

Our system addresses these challenges by integrating semantic information retrieval with deterministic academic decision-making.

The solution combines:

- **Natural Language Understanding:** Interprets student questions in Arabic and other supported language contexts.
- **Semantic Course Retrieval:** Identifies relevant courses using multilingual embeddings and vector similarity.
- **Academic Rule Validation:** Applies predefined academic constraints to course recommendations.
- **Controlled Language Generation:** Produces understandable responses based on retrieved information and validated decisions.

The project was developed collaboratively by a team of three students as a final-year graduation project at Al-Balqa Applied University.

---

## 2. Problem Statement

University students frequently face difficulties when planning their academic schedules.

Common challenges include:

- Understanding course prerequisites and academic requirements.
- Identifying appropriate courses for the upcoming semester.
- Checking eligibility based on completed credit hours.
- Understanding concurrent-course and elective restrictions.
- Finding accurate course information using natural language.
- Making registration decisions without continuous access to academic advisors.

Conventional information systems may provide course data without interpreting student questions or applying academic scheduling constraints.

**Our objective was to develop an intelligent academic assistant that combines flexible AI-based information retrieval with reliable, rule-based academic decision-making.**

---

## 3. System Architecture

The system follows a **Hybrid RAG Architecture** integrated with a deterministic academic scheduling engine.

### High-Level Workflow

```text
              Student Query
                    |
                    v
        Natural Language Processing
                    |
                    v
        Intent & Course Identification
                    |
          +---------+---------+
          |                   |
          v                   v
    Exact / Fuzzy       Semantic Search
       Matching          (E5 + FAISS)
          |                   |
          +---------+---------+
                    |
                    v
          Relevant Course Data
                    |
                    v
        Academic Rule Validation
                    |
                    v
         Validated Information
          / Recommendations
                    |
                    v
       Controlled Response Generation
             (Qwen2.5-7B)
                    |
                    v
           Student Interface
```

### 3.1 Semantic Retrieval Layer

The retrieval system uses the multilingual embedding model:

`intfloat/multilingual-e5-large`

The model transforms course descriptions and student queries into dense vector representations.

These representations enable semantic matching between student questions and academic information, even when the wording differs from the original course descriptions.

### 3.2 Vector Search Engine

**FAISS (Facebook AI Similarity Search)** is used for efficient similarity-based retrieval.

Its role includes:

- Indexing course embeddings.
- Searching for semantically relevant academic information.
- Supporting retrieval from structured course descriptions.

### 3.3 Multi-Stage Course Identification

The system combines multiple matching strategies:

1. **Exact Matching:** Identifies courses through direct name or identifier matches.
2. **Fuzzy Matching:** Handles minor spelling differences and approximate course names.
3. **Semantic Matching:** Uses embedding similarity to understand related academic terminology.

This approach improves the flexibility of course identification.

### 3.4 Rule-Based Academic Scheduling Engine

Academic recommendations are validated using deterministic rules.

The engine considers:

- Course prerequisites.
- Completed credit hours.
- Academic year eligibility.
- Concurrent-course requirements.
- Elective course limitations.

This separation helps prevent language-model output from overriding defined academic requirements.

### 3.5 Controlled Language Generation

The system uses:

`Qwen2.5-7B-Instruct`

The language model is responsible for formatting and presenting responses in understandable natural language.

Academic eligibility and scheduling decisions are determined by the rule-based system rather than relying exclusively on language-model generation.

### 3.6 Backend & Frontend

**Backend:**
- Python
- Flask
- REST APIs
- ngrok for development access

**Frontend:**
- HTML
- CSS
- JavaScript

The web interface allows students to interact with the academic assistant through a browser.

---

## 4. Key Features

### Intelligent Course Search

Retrieves relevant course descriptions and prerequisite information using exact, fuzzy, and semantic matching.

### Arabic-Language Academic Assistance

Supports natural-language interaction for academic questions, with multilingual semantic retrieval capabilities.

### Rule-Based Course Recommendations

Evaluates academic constraints before presenting semester course recommendations.

### Explainable Academic Decisions

Uses explicit academic rules to support understandable course eligibility decisions.

### NLP-Based Intent Detection

Identifies the purpose of student queries to guide the appropriate processing workflow.

### Automated Evaluation

Includes an evaluation framework for testing course identification, intent classification, answer accuracy, and response latency.

### Web-Based User Interface

Provides an accessible browser interface powered by Flask and frontend web technologies.

---

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Natural Language Processing | Sentence Transformers |
| Embedding Model | intfloat/multilingual-e5-large |
| Vector Search | FAISS |
| Large Language Model | Qwen2.5-7B-Instruct |
| Data Processing | Pandas, NumPy |
| APIs | REST APIs |
| Development Access | ngrok |
| Architecture | Hybrid RAG + Rule-Based Decision Making |

---

## 6. Evaluation & Testing

The system was evaluated using an automated testing framework involving hundreds of synthetic student queries.

The evaluation focused on the following metrics:

| Evaluation Metric | Purpose |
|---|---|
| Course Identification Accuracy | Measures how correctly the system identifies requested courses. |
| Intent Classification Accuracy | Evaluates the classification of student query intentions. |
| Direct-Answer Accuracy | Assesses correctness of responses to academic questions. |
| Response Latency | Measures the time required to generate responses. |

The evaluation framework was designed to examine the reliability of the retrieval and response-generation workflow.

**Note:** Numerical performance results are not included here because verified metric values have not been provided. Actual evaluation results should be added when available.

---

## 7. Project Structure

The repository includes the following main components:

```text
AI-Academic-Advisor-Chatbot/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- Evaluation
|-- statics/
|-- templates/
```

**Component descriptions:**

- `app.py` — Main Python application.
- `requirements.txt` — Python dependencies.
- `templates/` — Web interface templates.
- `statics/` — Frontend static resources.
- `Evaluation` — Project evaluation resource.
- `README.md` — Project documentation.

---

## 8. Installation & Setup

### Prerequisites

- Python
- pip
- Git
- Access to the models and resources required by the application

### Step 1: Clone the Repository

```bash
git clone https://github.com/azezsarisari/AI-Academic-Advisor-Chatbot.git
```

### Step 2: Navigate to the Project

```bash
cd AI-Academic-Advisor-Chatbot
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure and Run

Review `app.py` for the required model paths, datasets, configuration values, and application startup settings.

The exact execution procedure depends on the project's model-loading and development configuration.

**Note:** The setup instructions above describe the repository preparation steps. A fully verified end-to-end deployment guide requires testing the application and confirming its runtime dependencies.

---

## 9. Academic Significance

This project demonstrates the practical integration of multiple artificial intelligence and software engineering concepts:

- Retrieval-Augmented Generation.
- Multilingual semantic embeddings.
- Vector similarity search.
- Natural Language Processing.
- Deterministic decision systems.
- Large Language Model integration.
- Backend API development.
- Academic information retrieval.
- Automated AI system evaluation.

The project illustrates how AI-based retrieval and deterministic rule validation can work together in applications where reliability and explainability are important.

---

## 10. Graduation Project Team

This project was developed collaboratively by three students from **Al-Balqa Applied University**, as part of the Bachelor of Science program in Artificial Intelligence and Robotics.

### Team Members

**Abdalaziz Riyad Sari**
- GitHub: [azezsarisari](https://github.com/azezsarisari)
- LinkedIn: [Abdalaziz Sari](https://www.linkedin.com/in/abdalaziz-sari-b157b7306/)

**Ibnu Abi Hujayr (Al-Mugheeratu_Aws)**
- GitHub: [AlmugheeratuAws](https://github.com/AlmugheeratuAws)

**Hassan F. Ghuri**

### Team Collaboration

The project was completed as a collaborative academic effort.

Team members contributed to system design, development, implementation, testing, evaluation, and documentation.

Individual responsibilities are not separately attributed here because a verified task-by-task contribution breakdown has not been provided.

---

## 11. Original Repository & Attribution

This repository is an independently maintained portfolio copy of a collaborative university graduation project.

**Original Repository:**

[AI-Powered Academic Advisor — Original Team Repository](https://github.com/AlmugheeratuAws/AI-Powered-Academic-Advisor-Chatbot-For-semester-Course-Recommendation-in-Arabic-)

The original project was published under the GitHub account of team member **AlmugheeratuAws**.

This copy is maintained by **Abdalaziz Sari** for academic documentation, portfolio presentation, and professional reference.

All original team contributions and authorship are acknowledged. The original Git commit history is preserved.

---

## 12. Disclaimer

This project was developed for **academic and educational purposes**.

The academic rules, course information, and datasets are specific to the university context for which the system was originally designed.

Course recommendations should not be treated as official academic registration approval without verification against current university regulations.

---

## Contact

**Abdalaziz Sari**

AI Engineer | Machine Learning & NLP | RAG Systems & LLM Applications

- **GitHub:** [github.com/azezsarisari](https://github.com/azezsarisari)
- **LinkedIn:** [Abdalaziz Sari](https://www.linkedin.com/in/abdalaziz-sari-b157b7306/)
- **Email:** azezsarisari@gmail.com
- **Location:** Amman, Jordan

---

**Developed as a collaborative graduation project at Al-Balqa Applied University.**
