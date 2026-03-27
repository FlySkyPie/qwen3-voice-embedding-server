FROM python:3.12-bookworm

WORKDIR /app

RUN pip install pdm

COPY pyproject.toml pyproject.toml
COPY pdm.lock pdm.lock

RUN pdm install --prod

RUN apt-get update && apt-get install -y \
    libvulkan1 \
 && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8090

ENTRYPOINT [ "pdm", "run" ]

CMD ["start"]
