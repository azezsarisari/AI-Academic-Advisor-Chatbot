# AI-Powered-Academic-Advisor-Chatbot-For-semester-Course-Recommendation-in-Arabic
# Smart University Assistant – AI-Powered Academic Advising System

## Overview
The Smart University Assistant is an AI-powered academic advising system designed to help university students understand course information and make informed course registration decisions. The system combines semantic retrieval, deterministic rule-based logic, and controlled natural language generation to ensure accuracy, explainability, and academic reliability.

This project was developed as a **final-year graduation project** and implemented collaboratively as a **team effort**.

---

## Project Architecture
The system follows a hybrid **Retrieval-Augmented Generation (RAG)** architecture combined with rule-based decision making:

- **Semantic Retrieval Layer:** Uses the `intfloat/multilingual-e5-large` embedding model to convert course descriptions and student queries into dense vectors.
- **Vector Search Engine:** FAISS is used to efficiently retrieve semantically relevant courses.
- **Multi-Stage Course Identification:** Combines exact matching, fuzzy matching, and semantic search.
- **Rule-Based Scheduling Engine:** Enforces academic constraints such as prerequisites, completed credit hours, year eligibility, concurrent courses, and elective limits.
- **Language Generation Layer:** Uses `Qwen2.5-7B-Instruct` strictly for controlled natural language response formatting.
- **Frontend Interface:** Built using HTML, CSS, and JavaScript.
- **Backend API:** Implemented using Flask and exposed during development via ngrok.

---

## Key Features
- Accurate course description and prerequisite retrieval
- Explainable rule-based semester schedule recommendations
- Multilingual semantic understanding of student queries
- Lightweight NLP intent detection
- Automated evaluation and benchmarking framework
- Fully documented system architecture and methodology

---

## Technologies Used
- Python
- Flask
- Pandas, NumPy
- FAISS
- Sentence Transformers
- `intfloat/multilingual-e5-large`
- Qwen2.5-7B-Instruct
- HTML, CSS, JavaScript
- REST APIs, ngrok

---

## Evaluation
The system was evaluated using an automated testing framework on hundreds of synthetic student queries. Performance was measured across:
- Course identification accuracy
- Intent classification accuracy
- Direct-answer accuracy
- Response latency

---

## Team Contribution
This project was developed collaboratively by a team of three students.  
All members contributed to system design, implementation, testing, evaluation, and documentation.

**Team Members:**
- "Al-Mugheeratu_Aws" Ibnu Abi Hujayr
- Abdalaziz Riyad Sari
- Hassan F. Ghuri

---

## Disclaimer
This project is intended for academic and educational purposes only. All academic rules and datasets used are specific to the university context for which the system was developed.
