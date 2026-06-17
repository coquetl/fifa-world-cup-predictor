# ─────────────────────────────────────────────────────────
#  Dockerfile — Le Prédicteur de Jeff
#  Optimisé pour Google Cloud Run
# ─────────────────────────────────────────────────────────

# Image Python slim pour minimiser la taille du conteneur
FROM python:3.11-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Répertoire de travail
WORKDIR /app

# Copie des dépendances en premier (layer cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif
COPY app.py .
COPY templates/ ./templates/

# Copie du modèle pré-entraîné et des données
COPY model.joblib .
COPY data/ ./data/

# Port exposé (variable d'environnement injectée par Cloud Run)
EXPOSE $PORT

# Démarrage avec Gunicorn — production-ready
# Cloud Run injecte $PORT automatiquement
CMD exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 2 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    app:app
