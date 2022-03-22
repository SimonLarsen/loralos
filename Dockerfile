FROM python:3.8-slim-buster
EXPOSE 80

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY loralos/*.py loralos/config.ini /app/
COPY loralos/assets/* /app/assets/
COPY loralos/data/* /app/data/

ENTRYPOINT ["python"]
CMD ["app.py", "-p", "80"]
