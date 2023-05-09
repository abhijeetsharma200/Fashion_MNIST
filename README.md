# Fashion_MNIST

Using Docker to create an MLOps pipeline to predict on the Fashion MNIST Dataset.

To run the program manually first create a volume for persistent storage using:

```docker volume create my-volume```

Then going to each sub-directory and running the following commands which builds and runs the containers from the dockerfile. For example for data preparation step we have the following. 

# Data Preparation

```cd 1_data_preparation```

```docker build -t data_prep .```

```docker run -v my-volume:/app/data data_prep```

# Train

```docker build -t train .```

```docker run -v my-volume:/app/data train```

# Evaluate

```docker build -t eval .```

```docker run -v my-volume:/app/data eval```

# Deploy

```docker build -t deploy .```

```docker run -v my-volume:/app/data deploy```

# Orchestration

Using the current `docker-compose.yml` doesn't work properly as the order is not right resulting file not found errors.
