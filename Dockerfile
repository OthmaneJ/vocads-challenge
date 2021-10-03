FROM python:3.7-slim-buster
LABEL maintainer jebbariothmane2@gmail.com

WORKDIR /program

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY app.py .
COPY model/ .
COPY embeddings/ .

CMD ["python3", "./app.py"]
