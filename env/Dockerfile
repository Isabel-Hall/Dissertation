FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
RUN apt update
RUN apt install python3-pip -y
RUN pip install matplotlib numpy pandas scipy 
WORKDIR /app
CMD bash
