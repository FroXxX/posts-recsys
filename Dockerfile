FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY web_service/ /app/web_service/
COPY data/ /app/data/
COPY models/ /app/models/
# Копируем пакет TextPr в контейнер
COPY TextPreprocessor/ /app/TextPreprocessor/


RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app/"

CMD ["python3", "/app/web_service/src/main.py"]