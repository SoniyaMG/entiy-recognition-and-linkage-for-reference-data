FROM python:3.7.9-slim-buster

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

# Copy source code
COPY . /app

# expose the port number to outsiders
EXPOSE 5000

# Run application
CMD ["python","app.py"]