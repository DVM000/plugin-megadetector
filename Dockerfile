FROM python:3.10-slim

# Install OpenCV dependences
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/* 
 
# Download model
RUN mkdir -p /models
RUN wget -O /models/MDV6-yolov10x.pt https://zenodo.org/records/14567879/files/MDV6-yolov10x.pt?download=1

   
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
