FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# changing directory where main.py is located and then run both servers
CMD ["bash", "-c", "cd startup_app && uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run ../streamlit/app.py --server.port 8501"]
