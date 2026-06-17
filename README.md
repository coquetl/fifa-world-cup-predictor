# Predictor Coupe du Monde 2026 ⚽🏆

Ce projet est une application web Flask interactive permettant de simuler et prédire les résultats des matchs de la Coupe du Monde de la FIFA 2026.

## Fonctionnalités
- **Prédictions basées sur l'IA** : Modèle XGBoost robuste entraîné sur 10 ans d'historique de matchs internationaux réels (9 500+ matchs).
- **Mise à jour dynamique de l'Elo** : Les classements Elo des équipes se mettent à jour en temps réel à chaque score enregistré et se propagent automatiquement aux matchs suivants.
- **Forme chronologique & Repos** : Intègre les jours de repos et la forme actuelle des équipes dans les prédictions.
- **Phase finale dynamique** : Calcule automatiquement les classements de poule et propage les qualifiés (y compris les meilleurs troisièmes) dans l'arbre final.

## Structure du Projet
- `app.py` : Application Flask principale servant l'interface utilisateur et gérant les calculs de prédictions et d'Elos.
- `train_model.py` : Script d'entraînement du modèle de base XGBoost offline.
- `Dockerfile` : Fichier de configuration Docker pour le déploiement.
- `templates/index.html` : Interface web moderne et interactive.
- `data/` : Dossier contenant les données de base, le calendrier des matchs et les fichiers de résultats.

## Installation et Lancement

### Avec Docker
1. Construisez l'image Docker :
```bash
docker build -t fifa-world-cup-predictor .
```
2. Lancez le conteneur :
```bash
docker run -p 5000:5000 fifa-world-cup-predictor
```

### En local (Python)
1. Installez les dépendances :
```bash
pip install -r requirements.txt
```
2. Lancez l'application :
```bash
python app.py
```
Accédez à l'application via `http://127.0.0.1:5000`.
