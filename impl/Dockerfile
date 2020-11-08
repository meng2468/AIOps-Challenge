FROM python:3.6.8

WORKDIR /opt/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=true
CMD ["python", "consumer.py"]
