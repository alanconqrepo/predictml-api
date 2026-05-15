FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN addgroup --system streamlit && \
    adduser --system --ingroup streamlit streamlit && \
    chown -R streamlit:streamlit /app
EXPOSE 8501
USER streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.runOnSave=true", "--server.fileWatcherType=poll"]
