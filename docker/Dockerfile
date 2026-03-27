FROM python:3.12-bookworm

WORKDIR /app

RUN pip install pdm

COPY . .

RUN pdm install --prod

EXPOSE 8090

ENTRYPOINT [ "pdm", "run" ]

CMD ["start"]
