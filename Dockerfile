FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip install --no-cache-dir --prefer-binary -r requirements.txt
COPY . .
RUN python src/data_acquisition.py && python src/eda.py && python src/train.py
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
