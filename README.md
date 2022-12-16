# LoRaWAN line of sight helper ![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

A web tool for estimating LoRa line of sight in Denmark.

## Running

The application sources aerial photos and height maps from Danmarks Højdemodel.
Before running the application you must obtain an API key from [dataforsyningen.dk](https://dataforsyningen.dk) and set the environment variables `LORALOS_IMAGE_TOKEN` and `LORALOS_HEIGHTMAP_TOKEN`.

Install the necessary Python packages defined in `requirements.txt`:

```sh
pip install -r requirements.txt
```

Then start the dashboard by running the `app.py` script:

```sh
cd loralos
python app.py -p 8080
```

The Flask application is exposed at app:server. Example usage with gunicorn:

```sh
gunicorn -w 1 -b 0.0.0.0:8080 app:server
```

## Docker image

Docker images are available on Docker Hub under [SimonLarsen/loralos](https://hub.docker.com/repository/docker/simonlarsen/loralos). Example usage:

```sh
docker run -d -p 80:8080 \
    --env LORALOS_IMAGE_TOKEN=[APIKEY] \
    --env LORALOS_HEIGHTMAP_TOKEN=[APIKEY] \
    simonlarsen/loralos:latest
```
