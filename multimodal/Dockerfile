# DashInfer VLM (Vision-Language Model) container
# Based on NVIDIA PyTorch container with DashInfer + dashinfer-vlm

FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /root/

RUN mkdir -p /root/code/
COPY ./dashinfer_vlm /root/code/dashinfer_vlm
COPY ./setup.py code/
COPY ./requirements.txt /root/code/requirements.txt

# Install DashInfer v3.0.0
RUN python3 -m pip install --no-cache-dir \
    https://github.com/modelscope/dash-infer/releases/download/v3.0.0/dashinfer-3.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN python3 -m pip install --no-cache-dir -r /root/code/requirements.txt

RUN python3 -m pip install --no-cache-dir -e /root/code/

COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT  ["./entrypoint.sh"]
