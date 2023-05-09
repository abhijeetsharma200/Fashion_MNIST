FROM python:latest

RUN pip install numpy tensorflow flask

# Mount the volume to /app/data inside the container
VOLUME ["/app/data"]

# Copy the Flask app to the container
WORKDIR /app
COPY app.py /app

# Expose port 5000
EXPOSE 5000

CMD ["python", "app.py"]
