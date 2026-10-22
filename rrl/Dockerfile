# CPU image for FNN training: docker build -t faisaarlab/fnn2026-rrl:v1 rrl
FROM python:3.10.14-slim-bookworm
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt \
    && cd /usr/local/lib/python3.10/site-packages/torch \
    && rm -rf /tmp/requirements.txt include test bin/*Test bin/test_* bin/protoc* bin/tutorial_* \
              lib/libjitbackend_test.so lib/libtorchbind_test.so lib/libbackend_with_compiler.so \
    && find /usr/local/lib/python3.10/site-packages -type d -name tests -prune -exec rm -rf {} + \
    && pip uninstall -y -q pip setuptools wheel
ENV PYTHONDONTWRITEBYTECODE=1
