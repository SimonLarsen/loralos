FROM python:3.8-slim-buster
EXPOSE 8080

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY loralos/*.py loralos/config.ini /app/
COPY loralos/assets/* /app/assets/
COPY loralos/data/* /app/data/

ENTRYPOINT ["gunicorn"]
CMD ["-w", "1", "-b", "0.0.0.0:8080", "app:server"]
