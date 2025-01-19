# Use an official NVIDIA CUDA image as the base image (if using GPU)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install required packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opengl \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglu1-mesa \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libx11-dev \
    libxext-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace/3DShape2VecSet/FID/

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the application
CMD ["python3", "evaluate_kid.py"]
