# Pinned base so a rebuild in six months produces the same figures.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

ENV PYTHONPATH=/app/src MPLBACKEND=Agg

# Regenerate the characterisation. Mount a volume at /app/reports to keep it.
CMD ["python", "scripts/characterize.py"]
