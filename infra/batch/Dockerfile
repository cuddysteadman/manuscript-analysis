FROM tensorflow/tensorflow:2.16.1-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.lock /opt/manuscript_analysis/requirements.lock
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /opt/manuscript_analysis/requirements.lock

COPY src /opt/manuscript_analysis/src
COPY config /opt/manuscript_analysis/config
COPY pyproject.toml batch_entrypoint.py /opt/manuscript_analysis/
RUN python -m pip install --no-deps /opt/manuscript_analysis && \
    chmod 0555 /opt/manuscript_analysis/batch_entrypoint.py

ENTRYPOINT ["python", "/opt/manuscript_analysis/batch_entrypoint.py"]
