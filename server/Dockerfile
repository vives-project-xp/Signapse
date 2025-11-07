FROM python:3.12-slim

WORKDIR /app

COPY ./server/requirements.txt .

RUN apt-get update  && \
    apt-get install -y \
        curl \
        libgl1 \
        libglib2.0-0 \
        && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -q -r requirements.txt

COPY ./notebooks/package /package

RUN pip install -e /package

COPY ./server .

EXPOSE 8000

CMD ["fastapi","run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]
