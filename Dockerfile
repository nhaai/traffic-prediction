FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --upgrade pip
RUN pip install \
    ultralytics==8.1.0 \
    opencv-python==4.9.0.80 \
    flask==3.0.2 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.1 \
    matplotlib==3.8.2 \
    joblib==1.3.2
RUN mkdir -p static/uploads
EXPOSE 5000
CMD ["python", "app.py"]
