FROM python:3.9-slim

# Avoid interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive

# System deps for OpenCV + Graphviz
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Optionally set Flask env (nếu dùng app.py trực tiếp thì không cần)
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=5000

EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
