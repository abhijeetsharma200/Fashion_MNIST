FROM python:latest
RUN pip install numpy tensorflow

# Create the /app/data directory
RUN mkdir -p /app/data

WORKDIR /app
COPY data_preparation.py /app

# Mount a volume to the /app/data directory
VOLUME ["/app/data"]

CMD ["python", "data_preparation.py"]
