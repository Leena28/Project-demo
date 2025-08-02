# AI STARTUP ASSISTANT APP (RAG Chatbot & ML Valuation Predictor)

A full-stack AI assistant that combines retrieval-augmented generation (RAG) with machine learning to help startups evaluate funding potential and interactively explore business-relevant knowledge.

**Live Demo**: [http://51.21.255.6:8501/](http://51.21.255.6:8501/)  
**Tech Stack**: Python 3.10, FastAPI, Streamlit, Transformers, LangChain, FAISS, XGBoost,RAG 
**Deployment**: Dockerized on AWS EC2 (Ubuntu)

---

## PROJECT OVERVIEW

This application serves two purposes:

- A **valuation prediction tool** that estimates a startup's valuation based on funding amount, growth rate, industry, and geography using a trained XGBoost regressor.
- An **AI-powered startup chatbot** that leverages a FAISS-based RAG pipeline and Google's FLAN-T5-Small model to answer startup-related questions with precision.

All data is sourced from public startup platforms (ex-Crunchbase,TechCrunch,Y Combinator etc) and industry sources, and the system is deployed on a cloud server AWS using Docker Compose for separation of frontend and backend.
   
Chatbot snapshot-
<img width="1268" height="519" alt="chatbot_feature" src="https://github.com/user-attachments/assets/a774f790-e87a-48a8-aaef-9b8879d27977" />
Valuation Predictor-
<img width="1275" height="682" alt="predictor1" src="https://github.com/user-attachments/assets/d7dd488d-0692-42ea-b44f-1e3e0232f7c5" />
<img width="923" height="368" alt="predictor_2" src="https://github.com/user-attachments/assets/8b89182d-6d6d-4ce7-8eae-e06e74d7d7df" />



---

## KEY FEATURES

- **Valuation Prediction Engine** using XGBoost and engineered features
- **RAG Chatbot** powered by LangChain Refine Chain and GOOGLE FLAN-T5-Small
- **FAISS Semantic Search** over curated startup knowledge
- **Data-informed Default Growth Rates** per industry category
- **Post-prediction Valuation Capping** for realism (2× to 10× funding)
- **Deployed on EC2** with Docker, accessible publicly
- **End-to-End Pipeline** from data scraping,exploratory data analysis,data preprocessing,model building,deployment on cloud to interactive user interface

---

## TECH STACK & ARCHITECTURE

- **Frontend**: Streamlit (served via Docker)
- **Backend**: FastAPI, LangChain, FAISS, Transformers, HuggingFace Pipeline
- **ML Models**: XGBoost Regressor for valuation, FLAN-T5-Small for generation
- **Vector Embedding**: MiniLM from HuggingFace
- **Infrastructure**: AWS EC2, Docker Compose, UFW, exposed port via security group

DOCKER CONTAINERS-
  <img width="995" height="535" alt="docker_containers_running" src="https://github.com/user-attachments/assets/ed951646-80e9-4dd3-8bdf-e5c7ff0bcb5b" />

AWS EC2 INSTANCE-
<img width="1366" height="537" alt="AWS_INSTANCE" src="https://github.com/user-attachments/assets/521e16e2-a636-406e-90c4-d8c8bfab62c9" />

---

## INSTALLATION & QUICK START

1. Clone the repository
2. Build and run containers:
   docker-compose up -d

3. Access the application at:
   http://51.21.255.6:8501/

---
## DATA SOURCES & METHODOLOGY

### Valuation Model

* Data scrapped from: TechCrunch, Crunchbase, Pitchbook, GrowthList
* Applied robust preprocessing:

  * Outlier detection via IQR and Box-Cox transformations
  * Standardization with ZScaler
  * Applied Log Transformation on numerical columns
  * Engineered feature: log(funding)/growth rate ratio
  * Created normalized industry growth rate features (e.g. AI: 20%, Finance: 15%, Aerospace: 7.8%)

### RAG Chatbot

* Custom corpus built from:

  * Y Combinator startup guides
  * Brex and The Recursive blog posts
  * Exploding Topics and related startup education content
* Chunked and indexed via FAISS
* Retrieved context passed to LangChain RefineDocumentsChain and FLAN-T5 model

---

## API ENDPOINTS

* `POST /predict-valuation`
  Takes JSON payload with amount, industry, country, growth rate, and unit
  Returns predicted valuation

* `POST /ask-chatbot`
  Takes JSON query
  Returns AI-generated answer based on embedded documents context

---

## DEPLOYMENT

* Two separate Dockerfiles for backend and frontend
* Deployed using `docker-compose` in **detached mode**
* Verified containers running on EC2 (`docker ps`)
* Security groups configured to expose necessary ports (8000, 8501)
* UFW rules active on EC2 for traffic control

---
## IMPACT

This project showcases:

* Full-stack ML engineering: from data acquisition and cleaning to ML modeling and web deployment
* Retrieval-Augmented Generation (RAG) pipeline in production using LangChain and FAISS
* Real-world deployment on AWS with Docker and secured APIs
* Clear separation of concerns between ML logic, API serving, and frontend UX

---
## AUTHOR & CONTACT
Feel free to reach out:

* **LinkedIn**: *\[www.linkedin.com/in/leena-harpal-b4327a141]*
* **Email**: *\[leenaharpal96@gmail.com]*
