FROM python:latest
RUN pip install numpy tensorflow

# Mount the volume to /app/data inside the container
VOLUME ["/app/data"]

WORKDIR /app
COPY train.py /app

CMD ["python", "train.py"]
