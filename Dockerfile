FROM python:3.11.6

ENV GOOGLE_APPLICATION_CREDENTIALS "/app/service_account_key.json"

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "main.py"]
