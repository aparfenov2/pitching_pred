FROM python:3.10

ADD torch-1.13.0-cp310-cp310-manylinux1_x86_64.whl .
ADD nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl .
ADD nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl .
ADD nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl .

RUN pip install torch-1.13.0-cp310-cp310-manylinux1_x86_64.whl \
    nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl \
    nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl \
    nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl

ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
