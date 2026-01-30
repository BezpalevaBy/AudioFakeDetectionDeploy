FROM python:3.10-slim

COPY . /app
WORKDIR /app

# Копируем только requirements.txt сначала
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Устанавливаем PYTHONPATH
ENV PYTHONPATH=/app

RUN ls -la && \
    echo "=== app directory ===" && \
    ls -la app/

# Запускаем
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]