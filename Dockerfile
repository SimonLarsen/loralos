FROM python:3.8-slim-buster
EXPOSE 80

RUN mkdir /app
WORKDIR /app

ADD requirements.txt /app/
RUN pip install -r requirements.txt

COPY *.py config.ini stations.csv /app/
COPY assets/* /app/assets/

ENTRYPOINT ["python"]
CMD ["app.py", "-p", "80"]
