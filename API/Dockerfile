FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt install -y gcc pkg-config libhdf5-dev && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

ARG ENV=production
ENV ENV=${ENV}

COPY . .

RUN if [ "$ENV" != "local" ]; then \
  rm -rf /ml-models /ml-models/; \
  fi

CMD ["python3", "main.py"]