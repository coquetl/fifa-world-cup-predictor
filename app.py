"""
app.py — Match Result Predictor (Refactored)
Application Flask pour la Coupe du Monde 2026
"""

import os
import csv
import io
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from flask import Flask, render_template, request, jsonify, redirect, url_for
from functools import cmp_to_key

# ─────────────────────────────────────────────
# GOOGLE CLOUD STORAGE — couche de persistance
# ─────────────────────────────────────────────
# Si GCS_BUCKET_NAME est défini (Cloud Run), les fichiers mutables sont lus/écrits
# dans GCS. Sinon (développement local), on utilise le disque local comme avant.
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
_gcs_bucket = None

def _get_gcs_bucket():
    """Initialise et retourne le bucket GCS (singleton)."""
    global _gcs_bucket
    if _gcs_bucket is None and GCS_BUCKET_NAME:
        try:
            from google.cloud import storage
            client = storage.Client()
            _gcs_bucket = client.bucket(GCS_BUCKET_NAME)
            print(f"[GCS] Bucket connecté : {GCS_BUCKET_NAME}")
        except Exception as e:
            print(f"[GCS] Impossible de se connecter au bucket : {e}")
    return _gcs_bucket

def gcs_read_text(blob_name):
    """Lit un fichier texte depuis GCS. Retourne None si indisponible."""
    bucket = _get_gcs_bucket()
    if bucket:
        try:
            blob = bucket.blob(blob_name)
            return blob.download_as_text(encoding="utf-8-sig")
        except Exception as e:
            print(f"[GCS] Erreur lecture {blob_name} : {e}")
    return None

def gcs_write_text(blob_name, text_content):
    """Écrit une chaîne texte dans GCS."""
    bucket = _get_gcs_bucket()
    if bucket:
        try:
            blob = bucket.blob(blob_name)
            blob.upload_from_string(text_content, content_type="text/plain; charset=utf-8")
            print(f"[GCS] Fichier sauvegardé : {blob_name}")
            return True
        except Exception as e:
            print(f"[GCS] Erreur écriture {blob_name} : {e}")
    return False

def read_csv(local_path, blob_name, **kwargs):
    """Lit un CSV depuis GCS (si dispo) ou depuis le disque local."""
    if GCS_BUCKET_NAME:
        text = gcs_read_text(blob_name)
        if text is not None:
            return pd.read_csv(io.StringIO(text), **kwargs)
    return pd.read_csv(local_path, **kwargs)

def write_csv(df, local_path, blob_name, **kwargs):
    """Écrit un DataFrame en CSV vers GCS (si dispo) ET vers le disque local."""
    df.to_csv(local_path, **kwargs)
    if GCS_BUCKET_NAME:
        gcs_write_text(blob_name, df.to_csv(index=False))

def read_json(local_path, blob_name):
    """Lit un JSON depuis GCS (si dispo) ou depuis le disque local."""
    if GCS_BUCKET_NAME:
        text = gcs_read_text(blob_name)
        if text is not None:
            return json.loads(text)
    with open(local_path, encoding="utf-8") as f:
        return json.load(f)

def write_json(data, local_path, blob_name):
    """Écrit un JSON vers GCS (si dispo) ET vers le disque local."""
    text = json.dumps(data, ensure_ascii=False, indent=2)
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(text)
    if GCS_BUCKET_NAME:
        gcs_write_text(blob_name, text)

def read_results_csv(local_path, blob_name):
    """Lit le CSV des résultats saisis. Retourne [] si inexistant."""
    if GCS_BUCKET_NAME:
        text = gcs_read_text(blob_name)
        if text is not None:
            reader = csv.DictReader(io.StringIO(text))
            return list(reader)
    if os.path.exists(local_path):
        with open(local_path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    return []

def write_results_csv(rows, local_path, blob_name):
    """Écrit le CSV des résultats saisis vers GCS et disque local."""
    fieldnames = ["match_id", "home_score", "away_score", "result", "shootout_winner"]
    # Écriture locale
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Écriture GCS
    if GCS_BUCKET_NAME:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        gcs_write_text(blob_name, buf.getvalue())

# Noms des blobs GCS (chemins dans le bucket)
GCS_POULES_BLOB     = "data/matchs_poules.csv"
GCS_ARBRE_BLOB      = "data/arbre_phases_finales.json"
GCS_RESULTS_BLOB    = "data/coupe_du_monde_2026.csv"


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "predictions.csv")
REAL_RESULTS_PATH = os.path.join(BASE_DIR, "data", "coupe_du_monde_2026.csv")
POULES_PATH = os.path.join(BASE_DIR, "data", "matchs_poules.csv")
ARBRE_PATH = os.path.join(BASE_DIR, "data", "arbre_phases_finales.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

# Charger model.joblib au démarrage de Flask
print("[Flask] Chargement du package de modèle offline...")
model_data = joblib.load(MODEL_PATH)
base_model      = model_data["model"]
FEATURES        = model_data["features"]
LABEL_MAP       = model_data["label_map"]
LABEL_MAP_INV   = model_data["label_map_inv"]
X_train_hist    = model_data["X_train_hist"]
y_train_hist    = model_data["y_train_hist"]
player_map      = model_data["player_map"]
base_elo_ratings = model_data["base_elo_ratings"]
last_historical_match_dates = model_data.get("last_historical_match_dates", {})

# Traduction des clés Anglais -> Français pour la cohérence avec le calendrier
TEAM_TRANSLATION_FR_TO_EN = {
    'Mexique': 'Mexico', 'Afrique du Sud': 'South Africa', 'République de Corée': 'South Korea', 'Tchéquie': 'Czech Republic',
    'Canada': 'Canada', 'Bosnie-Herzégovine': 'Bosnia and Herzegovina', 'Qatar': 'Qatar', 'Suisse': 'Switzerland',
    'Brésil': 'Brazil', 'Maroc': 'Morocco', 'Haïti': 'Haiti', 'Écosse': 'Scotland',
    'États-Unis': 'United States', 'Paraguay': 'Paraguay', 'Australie': 'Australia', 'Turquie': 'Turkey',
    'Allemagne': 'Germany', 'Curaçao': 'Curaçao', 'Côte d’Ivoire': 'Ivory Coast', 'Équateur': 'Ecuador',
    'Pays-Bas': 'Netherlands', 'Japon': 'Japan', 'Suède': 'Sweden', 'Tunisie': 'Tunisia',
    'Belgique': 'Belgium', 'Égypte': 'Egypt', 'Iran': 'Iran', 'Nouvelle-Zélande': 'New Zealand',
    'Espagne': 'Spain', 'Cap Vert': 'Cape Verde', 'Arabie Saoudite': 'Saudi Arabia', 'Uruguay': 'Uruguay',
    'France': 'France', 'Sénégal': 'Senegal', 'Irak': 'Iraq', 'Norvège': 'Norway',
    'Argentine': 'Argentina', 'Algérie': 'Algeria', 'Autriche': 'Austria', 'Jordanie': 'Jordan',
    'Portugal': 'Portugal', 'RD Congo': 'DR Congo', 'Ouzbékistan': 'Uzbekistan', 'Colombie': 'Colombia',
    'Angleterre': 'England', 'Croatie': 'Croatia', 'Ghana': 'Ghana', 'Panama': 'Panama'
}
TEAM_TRANSLATION_EN_TO_FR = {v: k for k, v in TEAM_TRANSLATION_FR_TO_EN.items()}

# Convertir les dictionnaires pour utiliser les clés en français
base_elo_ratings = {TEAM_TRANSLATION_EN_TO_FR.get(k, k): v for k, v in base_elo_ratings.items()}
player_map = {TEAM_TRANSLATION_EN_TO_FR.get(k, k): v for k, v in player_map.items()}
last_historical_match_dates = {TEAM_TRANSLATION_EN_TO_FR.get(k, k): v for k, v in last_historical_match_dates.items()}

# Charger et prioritiser l'Elo baseline WC2026 (plus précis que l'Elo historique calculé)
BASELINE_PATH = os.path.join(BASE_DIR, "WC2026 Match Probability Baseline Dataset", "future_match_probabilities_baseline.csv")
try:
    baseline_df = pd.read_csv(BASELINE_PATH)
    for _, row in baseline_df.iterrows():
        if not pd.isna(row.get("home_elo")):
            fr_name = TEAM_TRANSLATION_EN_TO_FR.get(str(row["home_team"]), str(row["home_team"]))
            base_elo_ratings[fr_name] = float(row["home_elo"])
        if not pd.isna(row.get("away_elo")):
            fr_name = TEAM_TRANSLATION_EN_TO_FR.get(str(row["away_team"]), str(row["away_team"]))
            base_elo_ratings[fr_name] = float(row["away_elo"])
    print(f"[Flask] Elo baseline chargé — Mexique: {base_elo_ratings.get('Mexique', 'N/A')}, Afrique du Sud: {base_elo_ratings.get('Afrique du Sud', 'N/A')}")
except Exception as e:
    print(f"[Flask] Impossible de charger le baseline Elo : {e}")

def get_confederation(country_name):
    if not country_name or not isinstance(country_name, str):
        return 'OTHER'
    c = country_name.lower().strip()
    if c in {
        'usa', 'united states', 'mexico', 'mexique', 'canada', 'panama', 'haiti', 'haïti', 
        'curacao', 'curaçao', 'costa rica', 'honduras', 'jamaica', 'el salvador', 
        'trinidad and tobago', 'guatemala', 'cuba', 'martinique', 'guadeloupe'
    }:
        return 'CONCACAF'
    if c in {
        'brazil', 'brésil', 'argentina', 'argentine', 'colombia', 'colombie', 
        'uruguay', 'ecuador', 'équateur', 'paraguay', 'chile', 'peru', 'venezuela', 'bolivia'
    }:
        return 'CONMEBOL'
    if c in {
        'germany', 'allemagne', 'france', 'spain', 'espagne', 'england', 'angleterre', 
        'italy', 'italie', 'netherlands', 'pays-bas', 'belgium', 'belgique', 
        'portugal', 'croatia', 'croatie', 'switzerland', 'suisse', 'turkey', 'turquie', 
        'sweden', 'suède', 'norway', 'norvège', 'austria', 'autriche', 'scotland', 'écosse', 
        'czech republic', 'tchéquie', 'bosnia and herzegovina', 'bosnie-herzégovine', 
        'wales', 'ukraine', 'poland', 'denmark', 'russia', 'greece', 'serbia'
    }:
        return 'UEFA'
    if c in {
        'south africa', 'afrique du sud', 'morocco', 'maroc', 'senegal', 'sénégal', 
        'egypt', 'égypte', 'tunisia', 'tunisie', 'algeria', 'algérie', 'ghana', 
        'dr congo', 'rd congo', 'cape verde', 'cap vert', 'cabo verde', 'ivory coast', 
        'côte d’ivoire', 'côte d\'ivoire', 'nigeria', 'cameroon', 'mali', 'guinea'
    }:
        return 'CAF'
    if c in {
        'japan', 'japon', 'south korea', 'république de corée', 'korea republic', 
        'iran', 'ir iran', 'saudi arabia', 'arabie saoudite', 'qatar', 'iraq', 'irak', 
        'australia', 'australie', 'jordan', 'jordanie', 'uzbekistan', 'ouzbékistan', 
        'china', 'syria', 'vietnam', 'thailand', 'oman', 'india'
    }:
        return 'AFC'
    if c in {'new zealand', 'nouvelle-zélande', 'fiji', 'tahiti', 'solomon islands'}:
        return 'OFC'
    return 'OTHER'

print(f"[Flask] Modèle chargé, francisé avec dates historiques. Dataset historique : {X_train_hist.shape}")

# Dictionnaire pour les drapeaux des pays
FLAG_MAP = {
    "MEX":"🇲🇽","RSA":"🇿🇦","KOR":"🇰🇷","CAN":"🇨🇦","QAT":"🇶🇦",
    "SUI":"🇨🇭","BRA":"🇧🇷","MAR":"🇲🇦","HAI":"🇭🇹","SCO":"🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "USA":"🇺🇸","PAR":"🇵🇾","AUS":"🇦🇺","GER":"🇩🇪","CUR":"🇨🇼",
    "CIV":"🇨🇮","ECU":"🇪🇨","NED":"🇳🇱","JPN":"🇯🇵","TUN":"🇹🇳",
    "BEL":"🇧🇪","EGY":"🇪🇬","IRN":"🇮🇷","NZL":"🇳🇿","ESP":"🇪🇸",
    "CPV":"🇨🇻","KSA":"🇸🇦","URU":"🇺🇾","FRA":"🇫🇷","SEN":"🇸🇳",
    "NOR":"🇳🇴","ARG":"🇦🇷","ALG":"🇩🇿","AUT":"🇦🇹","JOR":"🇯🇴",
    "POR":"🇵🇹","UZB":"🇺🇿","COL":"🇨🇴","ENG":"🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "CRO":"🇭🇷","GHA":"🇬🇭","PAN":"🇵🇦","CZE":"🇨🇿","BIH":"🇧🇦",
    "TUR":"🇹🇷","SWE":"🇸🇪","IRQ":"🇮🇶","COD":"🇨🇩"
}

# player_stats supprimé : les ratings FotMob (std=0.15) étaient non-discriminants
# Le modèle utilise uniquement Elo, forme, repos et voyage.

def outcome(hs, as_):
    if hs > as_: return "W"
    elif hs == as_: return "D"
    return "L"

def exp_prob(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

# ─────────────────────────────────────────────
# TIEBREAKERS & STANDINGS
# ─────────────────────────────────────────────
def compare_teams(t1, t2):
    """Tri officiel : Points -> GD -> GF -> Wins -> Team Name (asc)"""
    if t1["points"] != t2["points"]:
        return t2["points"] - t1["points"]
    if t1["gd"] != t2["gd"]:
        return t2["gd"] - t1["gd"]
    if t1["gf"] != t2["gf"]:
        return t2["gf"] - t1["gf"]
    if t1["wins"] != t2["wins"]:
        return t2["wins"] - t1["wins"]
    if t1["team"] < t2["team"]:
        return -1
    elif t1["team"] > t2["team"]:
        return 1
    return 0

def solve_bipartite_matching(qualified_groups):
    """Backtracking d'affectation des 8 meilleurs troisièmes."""
    slots = {
        75: {'A', 'B', 'C', 'D', 'F'},
        78: {'C', 'D', 'F', 'G', 'H'},
        79: {'C', 'E', 'F', 'H', 'I'},
        80: {'E', 'H', 'I', 'J', 'K'},
        81: {'A', 'E', 'H', 'I', 'J'},
        82: {'B', 'E', 'F', 'I', 'J'},
        85: {'E', 'F', 'G', 'I', 'J'},
        88: {'D', 'E', 'I', 'J', 'L'}
    }
    match_ids = list(slots.keys())
    assignment = {}
    
    def backtrack(idx, remaining):
        if idx == len(match_ids):
            return True
        match_id = match_ids[idx]
        allowed = slots[match_id]
        for g in list(remaining):
            if g in allowed:
                assignment[match_id] = g
                next_remaining = remaining.copy()
                next_remaining.remove(g)
                if backtrack(idx + 1, next_remaining):
                    return True
                del assignment[match_id]
        return False
        
    if backtrack(0, qualified_groups):
        return assignment
    return None

def compute_group_standings(poules_matches):
    # Regrouper les équipes par groupe
    groups_teams = {}
    for m in poules_matches:
        g = m["group"]
        if g not in groups_teams:
            groups_teams[g] = set()
        groups_teams[g].add(m["home_team"])
        groups_teams[g].add(m["away_team"])
        
    standings = {}
    for g, teams in groups_teams.items():
        standings[g] = {t: {"team": t, "points": 0, "gf": 0, "ga": 0, "gd": 0, "wins": 0, "group": g} for t in teams}
        
    for m in poules_matches:
        g = m["group"]
        h = m["home_team"]
        a = m["away_team"]
        
        hs_val = m.get("home_score_real", "")
        as_val = m.get("away_score_real", "")
        
        if hs_val != "" and as_val != "":
            hs = int(hs_val)
            as_ = int(as_val)
            
            standings[g][h]["gf"] += hs
            standings[g][h]["ga"] += as_
            standings[g][a]["gf"] += as_
            standings[g][a]["ga"] += hs
            
            if hs > as_:
                standings[g][h]["points"] += 3
                standings[g][h]["wins"] += 1
            elif hs < as_:
                standings[g][a]["points"] += 3
                standings[g][a]["wins"] += 1
            else:
                standings[g][h]["points"] += 1
                standings[g][a]["points"] += 1

    sorted_standings = {}
    teams_info = init_calendar_teams_map()
    for g in standings:
        for t in standings[g]:
            standings[g][t]["gd"] = standings[g][t]["gf"] - standings[g][t]["ga"]
            code = teams_info.get(t, {}).get("code", "")
            standings[g][t]["flag"] = FLAG_MAP.get(code, "🏳️")
            standings[g][t]["code"] = code
        sorted_standings[g] = sorted(standings[g].values(), key=cmp_to_key(compare_teams))
        
    return sorted_standings

# ─────────────────────────────────────────────
# INSIGHT GENERATION (LE TIPS DE LA COQS)
# ─────────────────────────────────────────────
def generate_tips(home_name, away_name, h_elo, a_elo, p_home, p_draw, p_away, h_rating, h_speed, a_rating, a_speed):
    # Victoire écrasante (>75%)
    if p_home > 75.0:
        return f"L'IA est formelle, c'est un carnage programmé pour {home_name}. Tu peux parier ton PEL les yeux fermés."
    if p_away > 75.0:
        return f"L'IA est formelle, c'est un carnage programmé pour {away_name}. Tu peux parier ton PEL les yeux fermés."
    # Match nul / Purge
    if p_draw > 35.0:
        return "Le modèle prédit une purge tactique monumentale. Idéal pour faire une sieste de 90 minutes ensemble."
    # Écart Elo minuscule (<30)
    if abs(h_elo - a_elo) < 30:
        return "Match ultra-serré. Ça va se jouer sur un coup de pied arrêté ou une boulette du gardien, prépare les pop-corn."
    # Fallback
    if p_home > p_away:
        return f"Avantage {home_name} — le modèle anticipe une maîtrise collective."
    else:
        return f"Avantage {away_name} — favori clair pour s'imposer."

# ─────────────────────────────────────────────
# RETRAIN MODEL DYNAMICALLY
# ─────────────────────────────────────────────
def retrain_model_dynamic(played_matches):
    # Désactivé pour stabiliser le modèle et éviter le surapprentissage (overfitting)
    # sur un petit échantillon de matchs WC2026.
    # L'Elo se met toujours à jour chronologiquement pour chaque match.
    return base_model


def parse_match_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%d/%m/%Y %H:%M")
    except Exception:
        try:
            return pd.to_datetime(date_str)
        except Exception:
            return None

# ─────────────────────────────────────────────
# CORE RESOLVER
# ─────────────────────────────────────────────
def load_predictions():
    # 1. Charger matchs poules et arbre
    if not GCS_BUCKET_NAME and (not os.path.exists(POULES_PATH) or not os.path.exists(ARBRE_PATH)):
        return [], [], {}, {"total": 104, "played": 0, "correct": 0, "accuracy": 0}
        
    poules_df = read_csv(POULES_PATH, GCS_POULES_BLOB)
    poules_matches = poules_df.to_dict("records")
    
    arbre_matches = read_json(ARBRE_PATH, GCS_ARBRE_BLOB)
        
    # 2. Charger les vrais résultats saisis
    real_results = {}
    try:
        rows_raw = read_results_csv(REAL_RESULTS_PATH, GCS_RESULTS_BLOB)
        for row in rows_raw:
            mid = int(row["match_id"])
            real_results[mid] = {
                "home_score": str(row.get("home_score", "")).strip(),
                "away_score": str(row.get("away_score", "")).strip(),
                "result": str(row.get("result", "")).strip(),
                "shootout_winner": str(row.get("shootout_winner", "")).strip()
            }
    except Exception as e:
        print(f"[WARN] Erreur lecture résultats réels : {e}")

    # 3. Recalculer Elo et Forme chronologiquement (match_id 1 à 72 d'abord)
    current_elo = base_elo_ratings.copy()
    form_stats = {} # team -> {"points": 0, "played": 0}
    current_last_match_dates = last_historical_match_dates.copy()
    
    played_2026 = []
    
    # Remplir d'abord les poules
    for m in poules_matches:
        mid = int(m["match_id"])
        ht = m["home_team"]
        at = m["away_team"]
        
        m["home_elo_new"] = ""
        m["away_elo_new"] = ""
        
        # Mettre à jour Elo de départ pour ce match dans le dict
        m["home_elo"] = round(current_elo.get(ht, 1500.0), 0)
        m["away_elo"] = round(current_elo.get(at, 1500.0), 0)
        
        # Jours de repos
        m_date = parse_match_date(m["date"])
        h_prev = current_last_match_dates.get(ht)
        a_prev = current_last_match_dates.get(at)
        h_rest = (m_date - h_prev).days if h_prev and m_date else 10.0
        a_rest = (m_date - a_prev).days if a_prev and m_date else 10.0
        m["home_rest"] = float(min(h_rest, 10.0))
        m["away_rest"] = float(min(a_rest, 10.0))
        
        if m_date:
            current_last_match_dates[ht] = m_date
            current_last_match_dates[at] = m_date
            
        # Facteur de voyage
        m["home_travel"] = 1.0 if get_confederation(ht) == "CONCACAF" else 0.0
        m["away_travel"] = 1.0 if get_confederation(at) == "CONCACAF" else 0.0
        
        # Forme
        h_stats = form_stats.get(ht, {"points": 0, "played": 0})
        a_stats = form_stats.get(at, {"points": 0, "played": 0})
        
        m["home_form"] = h_stats["points"] / h_stats["played"] if h_stats["played"] > 0 else 0.0
        m["away_form"] = a_stats["points"] / a_stats["played"] if a_stats["played"] > 0 else 0.0
        
        if mid in real_results and real_results[mid]["home_score"] != "" and real_results[mid]["away_score"] != "":
            res = real_results[mid]
            hs = int(res["home_score"])
            as_ = int(res["away_score"])
            m["home_score_real"] = hs
            m["away_score_real"] = as_
            m["result_real"] = res["result"]
            m["shootout_winner"] = res["shootout_winner"]
            
            # Recalculer Elo
            eh, ea = current_elo.get(ht, 1500.0), current_elo.get(at, 1500.0)
            sa = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            sb = 1.0 - sa
            eh_new = eh + 32 * (sa - exp_prob(eh, ea))
            ea_new = ea + 32 * (sb - exp_prob(ea, eh))
            current_elo[ht] = eh_new
            current_elo[at] = ea_new
            m["home_elo_new"] = round(eh_new, 0)
            m["away_elo_new"] = round(ea_new, 0)
            
            # Mettre à jour Forme
            h_pts = 3 if hs > as_ else (1 if hs == as_ else 0)
            a_pts = 3 if hs < as_ else (1 if hs == as_ else 0)
            form_stats[ht] = {"points": h_stats["points"] + h_pts, "played": h_stats["played"] + 1}
            form_stats[at] = {"points": a_stats["points"] + a_pts, "played": a_stats["played"] + 1}
            
            # Ajouter aux joués pour réentraînement
            played_2026.append(m)
        else:
            m["home_score_real"] = ""
            m["away_score_real"] = ""
            m["result_real"] = ""
            m["shootout_winner"] = ""

    # Calculer les classements de groupes
    standings = compute_group_standings(poules_matches)
    
    # Vérifier si la phase de poule est finie
    group_stage_finished = (len([m for m in poules_matches if m["result_real"] != ""]) == 72)
    
    # 4. RÉSOUDRE L'ARBRE DE PHASE FINALE
    if group_stage_finished:
        # Trouver les 1ers et 2es de groupe
        group_winners = {} # '1A' -> 'Mexique' etc.
        third_placed_teams = []
        
        for g_letter, g_stand in standings.items():
            if len(g_stand) >= 1: group_winners[f"1{g_letter}"] = g_stand[0]["team"]
            if len(g_stand) >= 2: group_winners[f"2{g_letter}"] = g_stand[1]["team"]
            if len(g_stand) >= 3:
                third_team = g_stand[2]
                third_team["group"] = g_letter
                third_placed_teams.append(third_team)
                
        # Repêchage des 8 meilleurs troisièmes
        sorted_thirds = sorted(third_placed_teams, key=cmp_to_key(compare_teams))
        best_eight_thirds = sorted_thirds[:8]
        best_eight_groups = [t["group"] for t in best_eight_thirds]
        
        # Backtracking
        assignment = solve_bipartite_matching(best_eight_groups)
        
        # Mappage des 16es de finale (matches 73 à 88)
        for m in arbre_matches:
            mid = int(m["match_id"])
            if 73 <= mid <= 88:
                h_pl = m["home_placeholder"]
                a_pl = m["away_placeholder"]
                
                # Remplir le Home
                if h_pl in group_winners:
                    m["home_team"] = group_winners[h_pl]
                elif h_pl.startswith("3"):
                    # C'est un slot de 3e place (ex: 3ABCDF pour match 75)
                    # Trouver quel groupe qualifié a été assigné à ce match
                    g_assigned = assignment.get(mid, None) if assignment else None
                    if g_assigned:
                        m["home_team"] = next(t["team"] for t in best_eight_thirds if t["group"] == g_assigned)
                        m["home_placeholder"] = f"3{g_assigned}"
                        
                # Remplir le Away
                if a_pl in group_winners:
                    m["away_team"] = group_winners[a_pl]
                elif a_pl.startswith("3"):
                    g_assigned = assignment.get(mid, None) if assignment else None
                    if g_assigned:
                        m["away_team"] = next(t["team"] for t in best_eight_thirds if t["group"] == g_assigned)
                        m["away_placeholder"] = f"3{g_assigned}"
                        
        # Propagation récursive dans l'arbre pour les matchs joués
        for m in arbre_matches:
            mid = int(m["match_id"])
            ht = m["home_team"]
            at = m["away_team"]
            
            # S'assurer que les codes et flags sont renseignés si les équipes sont là
            if ht:
                team_data = next((v for k, v in player_map.items() if k == ht), None)
                # Sinon on cherche dans FLAG_MAP
                m["home_code"] = ht[:3].upper()
                # Trouver le code FIFA officiel si présent
                for name_fr, data in init_calendar_teams_map().items():
                    if name_fr == ht:
                        m["home_code"] = data["code"]
                m["home_flag"] = FLAG_MAP.get(m["home_code"], "🏳️")
            if at:
                m["away_code"] = at[:3].upper()
                for name_fr, data in init_calendar_teams_map().items():
                    if name_fr == at:
                        m["away_code"] = data["code"]
                m["away_flag"] = FLAG_MAP.get(m["away_code"], "🏳️")
                
            m["home_elo_new"] = ""
            m["away_elo_new"] = ""
            
            # Elo et Forme
            m["home_elo"] = round(current_elo.get(ht, 1500.0), 0) if ht else 1500.0
            m["away_elo"] = round(current_elo.get(at, 1500.0), 0) if at else 1500.0
            
            # Jours de repos et Voyage
            if ht and at:
                m_date = parse_match_date(m["date"])
                h_prev = current_last_match_dates.get(ht)
                a_prev = current_last_match_dates.get(at)
                h_rest = (m_date - h_prev).days if h_prev and m_date else 10.0
                a_rest = (m_date - a_prev).days if a_prev and m_date else 10.0
                m["home_rest"] = float(min(h_rest, 10.0))
                m["away_rest"] = float(min(a_rest, 10.0))
                
                if m_date:
                    current_last_match_dates[ht] = m_date
                    current_last_match_dates[at] = m_date
                    
                m["home_travel"] = 1.0 if get_confederation(ht) == "CONCACAF" else 0.0
                m["away_travel"] = 1.0 if get_confederation(at) == "CONCACAF" else 0.0
            else:
                m["home_rest"] = 10.0
                m["away_rest"] = 10.0
                m["home_travel"] = 0.0
                m["away_travel"] = 0.0
            
            h_stats = form_stats.get(ht, {"points": 0, "played": 0}) if ht else {"points": 0, "played": 0}
            a_stats = form_stats.get(at, {"points": 0, "played": 0}) if at else {"points": 0, "played": 0}
            m["home_form"] = h_stats["points"] / h_stats["played"] if h_stats["played"] > 0 else 0.0
            m["away_form"] = a_stats["points"] / a_stats["played"] if a_stats["played"] > 0 else 0.0

            # Si le match est joué
            if mid in real_results and real_results[mid]["home_score"] != "" and real_results[mid]["away_score"] != "":
                res = real_results[mid]
                hs = int(res["home_score"])
                as_ = int(res["away_score"])
                m["home_score_real"] = hs
                m["away_score_real"] = as_
                m["result_real"] = res["result"]
                m["shootout_winner"] = res["shootout_winner"]
                
                # Qui est le gagnant et le perdant ?
                winner = ""
                loser = ""
                if hs > as_:
                    winner = ht
                    loser = at
                elif hs < as_:
                    winner = at
                    loser = ht
                else:
                    # Nul - utiliser shootout_winner
                    sw = res["shootout_winner"]
                    if sw == "home" or sw == ht:
                        winner = ht
                        loser = at
                    else:
                        winner = at
                        loser = ht
                        
                # Recalculer Elo
                if ht and at:
                    eh, ea = current_elo.get(ht, 1500.0), current_elo.get(at, 1500.0)
                    sa = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
                    sb = 1.0 - sa
                    eh_new = eh + 32 * (sa - exp_prob(eh, ea))
                    ea_new = ea + 32 * (sb - exp_prob(ea, eh))
                    current_elo[ht] = eh_new
                    current_elo[at] = ea_new
                    m["home_elo_new"] = round(eh_new, 0)
                    m["away_elo_new"] = round(ea_new, 0)
                    
                    # Forme
                    h_pts = 3 if hs > as_ else (1 if hs == as_ else 0)
                    a_pts = 3 if hs < as_ else (1 if hs == as_ else 0)
                    form_stats[ht] = {"points": h_stats["points"] + h_pts, "played": h_stats["played"] + 1}
                    form_stats[at] = {"points": a_stats["points"] + a_pts, "played": a_stats["played"] + 1}
                    
                    played_2026.append(m)

                # Propager
                next_mid = m.get("next_match_id")
                if next_mid:
                    next_m = next((x for x in arbre_matches if int(x["match_id"]) == int(next_mid)), None)
                    if next_m:
                        if m.get("is_home_in_next") == True:
                            next_m["home_team"] = winner
                        else:
                            next_m["away_team"] = winner
                            
                # Gérer le perdant des demi-finales (101 et 102) vers le match de 3e place (103)
                loser_next_mid = m.get("loser_next_match_id")
                if loser_next_mid:
                    loser_m = next((x for x in arbre_matches if int(x["match_id"]) == int(loser_next_mid)), None)
                    if loser_m:
                        if m.get("is_loser_home_in_next") == True:
                            loser_m["home_team"] = loser
                        else:
                            loser_m["away_team"] = loser
            else:
                m["home_score_real"] = ""
                m["away_score_real"] = ""
                m["result_real"] = ""
                m["shootout_winner"] = ""

    # 5. RE-ENTRAÎNEMENT FLASH IA
    dynamic_model = retrain_model_dynamic(played_2026)
    
    # 6. CALCULER LES PRÉDICTIONS POUR TOUS LES MATCHS
    all_combined = poules_matches + arbre_matches
    for m in all_combined:
        mid = int(m["match_id"])
        ht = m.get("home_team", "")
        at = m.get("away_team", "")

        # Exposer is_knockout au template pour l'affichage conditionnel
        m["is_knockout"] = mid >= 73

        if not ht or not at:
            # Match de phase finale non déterminé
            m["p_home_win"] = 0.0
            m["p_draw"] = 0.0
            m["p_away_win"] = 0.0
            m["insight"] = "En attente des qualifications de la phase précédente."
            m["correct_prediction"] = ""
            continue
            
        # Calculer les features à l'instant T
        eh = current_elo.get(ht, 1500.0)
        ea = current_elo.get(at, 1500.0)
        
        h_stats = form_stats.get(ht, {"points": 0, "played": 0})
        a_stats = form_stats.get(at, {"points": 0, "played": 0})
        h_form = h_stats["points"] / h_stats["played"] if h_stats["played"] > 0 else 0.0
        a_form = a_stats["points"] / a_stats["played"] if a_stats["played"] > 0 else 0.0
        form_diff = h_form - a_form
        
        is_ko = 1.0 if mid >= 73 else 0.0
        
        h_rest = m.get("home_rest", 10.0)
        a_rest = m.get("away_rest", 10.0)
        h_travel = m.get("home_travel", 0.0)
        a_travel = m.get("away_travel", 0.0)
        
        # Utiliser les prédictions figées si le match est joué et a des valeurs valides
        p_home = m.get("pred_home_win", "")
        p_draw = m.get("pred_draw", "")
        p_away = m.get("pred_away_win", "")
        
        if m.get("result_real", "") != "" and p_home != "" and p_home is not None and not pd.isna(p_home):
            m["p_home_win"] = round(float(p_home), 1)
            m["p_draw"]     = round(float(p_draw), 1)
            m["p_away_win"] = round(float(p_away), 1)
        else:
            feat = np.array([[eh, ea, eh - ea, 1.0, is_ko, h_form, a_form, form_diff, h_rest, a_rest, h_travel, a_travel]], dtype=np.float32)
            dm = xgb.DMatrix(feat, feature_names=FEATURES)
            prob = dynamic_model.predict(dm)[0]
            
            m["p_home_win"] = round(float(prob[2]) * 100, 1)
            m["p_draw"]     = round(float(prob[1]) * 100, 1)
            m["p_away_win"] = round(float(prob[0]) * 100, 1)
        
        # Génération du tips
        m["insight"] = generate_tips(
            ht, at, eh, ea, m["p_home_win"], m["p_draw"], m["p_away_win"],
            None, None, None, None
        )
        
        if m.get("result_real", "") != "":
            if mid >= 73:
                # Phases finales : on compare uniquement p_home_win et p_away_win (pas de match nul)
                predicted = "W" if m["p_home_win"] >= m["p_away_win"] else "L"
            else:
                max_prob = max(m["p_home_win"], m["p_draw"], m["p_away_win"])
                predicted = "W" if max_prob == m["p_home_win"] else ("D" if max_prob == m["p_draw"] else "L")
            m["correct_prediction"] = "✓" if predicted == m["result_real"] else "✗"
        else:
            m["correct_prediction"] = ""

    # Séparer à nouveau
    group_matches_final = [m for m in all_combined if int(m["match_id"]) <= 72]
    arbre_matches_final = [m for m in all_combined if int(m["match_id"]) >= 73]
    
    # 7. METRIQUES GLOBALES D'ACCURACY
    played_all = [m for m in all_combined if m["result_real"] != ""]
    if not played_all:
        stats = {"total": len(all_combined), "played": 0, "correct": 0, "accuracy": 0}
    else:
        correct = sum(1 for m in played_all if m.get("correct_prediction") == "✓")
        stats = {
            "total": len(all_combined),
            "played": len(played_all),
            "correct": correct,
            "accuracy": round(correct / len(played_all) * 100, 1)
        }

    return group_matches_final, arbre_matches_final, standings, stats, current_elo, base_elo_ratings

def init_calendar_teams_map():
    # Retourne le même dictionnaire d'équipes utilisé dans init_calendar.py
    return {
        "Mexique": {"code": "MEX"}, "Afrique du Sud": {"code": "RSA"}, "République de Corée": {"code": "KOR"}, "Tchéquie": {"code": "CZE"},
        "Canada": {"code": "CAN"}, "Bosnie-Herzégovine": {"code": "BIH"}, "Qatar": {"code": "QAT"}, "Suisse": {"code": "SUI"},
        "Brésil": {"code": "BRA"}, "Maroc": {"code": "MAR"}, "Haïti": {"code": "HAI"}, "Écosse": {"code": "SCO"},
        "États-Unis": {"code": "USA"}, "Paraguay": {"code": "PAR"}, "Australie": {"code": "AUS"}, "Turquie": {"code": "TUR"},
        "Allemagne": {"code": "GER"}, "Curaçao": {"code": "CUR"}, "Côte d’Ivoire": {"code": "CIV"}, "Équateur": {"code": "ECU"},
        "Pays-Bas": {"code": "NED"}, "Japon": {"code": "JPN"}, "Suède": {"code": "SWE"}, "Tunisie": {"code": "TUN"},
        "Belgique": {"code": "BEL"}, "Égypte": {"code": "EGY"}, "Iran": {"code": "IRN"}, "Nouvelle-Zélande": {"code": "NZL"},
        "Espagne": {"code": "ESP"}, "Cap Vert": {"code": "CPV"}, "Arabie Saoudite": {"code": "KSA"}, "Uruguay": {"code": "URU"},
        "France": {"code": "FRA"}, "Sénégal": {"code": "SEN"}, "Irak": {"code": "IRQ"}, "Norvège": {"code": "NOR"},
        "Argentine": {"code": "ARG"}, "Algérie": {"code": "ALG"}, "Autriche": {"code": "AUT"}, "Jordanie": {"code": "JOR"},
        "Portugal": {"code": "POR"}, "RD Congo": {"code": "COD"}, "Ouzbékistan": {"code": "UZB"}, "Colombie": {"code": "COL"},
        "Angleterre": {"code": "ENG"}, "Croatie": {"code": "CRO"}, "Ghana": {"code": "GHA"}, "Panama": {"code": "PAN"}
    }

# ─────────────────────────────────────────────
# ROUTES FLASK
# ─────────────────────────────────────────────
def get_global_ranking(poules_matches, arbre_matches, current_elo, base_elo_ratings):
    teams_info = init_calendar_teams_map()
    all_teams = list(teams_info.keys())
    
    status_map = {t: "En lice (Poules)" for t in all_teams}
    
    # Check if group stage is finished
    group_played = [m for m in poules_matches if m.get("result_real", "") != ""]
    group_finished = (len(group_played) == 72)
    
    if group_finished:
        r32_teams = set()
        for m in arbre_matches:
            mid = int(m["match_id"])
            if 73 <= mid <= 88:
                if m.get("home_team"): r32_teams.add(m["home_team"])
                if m.get("away_team"): r32_teams.add(m["away_team"])
        
        for t in all_teams:
            if t not in r32_teams:
                status_map[t] = "Éliminé (Poules)"
            else:
                status_map[t] = "En lice (16es)"
                
    for m in sorted(arbre_matches, key=lambda x: int(x["match_id"])):
        mid = int(m["match_id"])
        ht = m.get("home_team", "")
        at = m.get("away_team", "")
        res = m.get("result_real", "")
        sw = m.get("shootout_winner", "")
        
        if not ht or not at or res == "":
            stage = m["stage"]
            if ht and status_map[ht].startswith("En lice"):
                status_map[ht] = f"En lice ({stage})"
            if at and status_map[at].startswith("En lice"):
                status_map[at] = f"En lice ({stage})"
            continue
            
        hs = int(m["home_score_real"])
        as_ = int(m["away_score_real"])
        
        winner = ""
        loser = ""
        if hs > as_:
            winner = ht
            loser = at
        elif hs < as_:
            winner = at
            loser = ht
        else:
            if sw == "home" or sw == ht:
                winner = ht
                loser = at
            else:
                winner = at
                loser = ht
                
        if 73 <= mid <= 88:
            status_map[loser] = "Éliminé (16es)"
            status_map[winner] = "En lice (8es)"
        elif 89 <= mid <= 96:
            status_map[loser] = "Éliminé (8es)"
            status_map[winner] = "En lice (Quarts)"
        elif 97 <= mid <= 100:
            status_map[loser] = "Éliminé (Quarts)"
            status_map[winner] = "En lice (Demis)"
        elif 101 <= mid <= 102:
            status_map[loser] = "En lice (3e place)"
            status_map[winner] = "En lice (Finale)"
        elif mid == 103:
            status_map[winner] = "🥉 3ème place"
            status_map[loser] = "4ème place"
        elif mid == 104:
            status_map[winner] = "🏆 Vainqueur"
            status_map[loser] = "🥈 Finaliste"
            
    ranking = []
    for t in all_teams:
        code = teams_info[t]["code"]
        flag = FLAG_MAP.get(code, "🏳️")
        start_elo = base_elo_ratings.get(t, 1500.0)
        curr_elo = current_elo.get(t, 1500.0)
        diff = curr_elo - start_elo
        
        ranking.append({
            "team": t,
            "code": code,
            "flag": flag,
            "start_elo": round(start_elo, 0),
            "current_elo": round(curr_elo, 0),
            "diff": round(diff, 0),
            "status": status_map[t]
        })
        
    ranking = sorted(ranking, key=lambda x: x["current_elo"], reverse=True)
    return ranking

@app.route("/")
def index():
    poules, arbre, standings, stats, current_elo, base_elo_ratings = load_predictions()
    
    # Organiser les matchs de poules par groupe pour l'affichage
    groups_matches = {}
    for m in poules:
        g = m["group"]
        if g not in groups_matches:
            groups_matches[g] = []
        groups_matches[g].append(m)
    groups_matches = dict(sorted(groups_matches.items()))
    
    # Classer les matchs de phase finale chronologiquement (par date/match_id)
    arbre_sorted = sorted(arbre, key=lambda x: int(x["match_id"]))
    
    global_ranking = get_global_ranking(poules, arbre, current_elo, base_elo_ratings)
    
    return render_template(
        "index.html",
        groups_matches=groups_matches,
        arbre_matches=arbre_sorted,
        standings=standings,
        stats=stats,
        current_elo=current_elo,
        base_elo_ratings=base_elo_ratings,
        global_ranking=global_ranking
    )

@app.route("/update_result", methods=["POST"])
def update_result():
    match_id = request.form.get("match_id", "").strip()
    home_score = request.form.get("home_score", "").strip()
    away_score = request.form.get("away_score", "").strip()
    shootout_winner = request.form.get("shootout_winner", "").strip() # nom d'équipe, 'home' ou 'away'

    if not match_id or home_score == "" or away_score == "":
        return jsonify({"status": "error", "message": "Données manquantes"}), 400

    try:
        hs = int(home_score)
        as_ = int(away_score)
        mid = int(match_id)
    except ValueError:
        return jsonify({"status": "error", "message": "Scores invalides"}), 400

    # Déterminer le résultat
    if hs > as_:
        result = "W"
    elif hs < as_:
        result = "L"
    else:
        # Match nul
        if mid >= 73:
            # Phase finale : TAB obligatoire.
            # shootout_winner peut être : "home", "away", ou le nom de l'équipe (envoyé par le frontend).
            # On résout en comparant avec les noms des équipes dans l'arbre.
            if not shootout_winner:
                return jsonify({"status": "error", "message": "Vainqueur de TAB manquant pour la phase finale"}), 400

            # Récupérer les noms d'équipes pour ce match depuis l'arbre
            arbre_data = read_json(ARBRE_PATH, GCS_ARBRE_BLOB)
            arbre_match = next((x for x in arbre_data if int(x["match_id"]) == mid), None)
            home_team_name = arbre_match["home_team"] if arbre_match else ""
            away_team_name = arbre_match["away_team"] if arbre_match else ""

            # Résoudre W/L selon la forme reçue ("home"/"away" ou nom d'équipe)
            if shootout_winner == "home" or (home_team_name and shootout_winner == home_team_name):
                result = "W"
                shootout_winner = home_team_name or "home"  # normaliser vers le nom d'équipe
            elif shootout_winner == "away" or (away_team_name and shootout_winner == away_team_name):
                result = "L"
                shootout_winner = away_team_name or "away"  # normaliser vers le nom d'équipe
            else:
                return jsonify({"status": "error", "message": f"Vainqueur de TAB invalide : '{shootout_winner}'"}), 400
        else:
            result = "D"

    # Figer la prédiction : calculer la prédiction live AVANT d'enregistrer le score et de réentraîner le modèle
    poules, arbre, _, _, _, _ = load_predictions()
    all_m = poules + arbre
    match_data = next((x for x in all_m if int(x["match_id"]) == mid), None)
    
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    if match_data:
        p_home = match_data["p_home_win"]
        p_draw = match_data["p_draw"]
        p_away = match_data["p_away_win"]

    # Sauvegarder la prédiction dans le CSV ou JSON correspondant
    if mid <= 72:
        df = read_csv(POULES_PATH, GCS_POULES_BLOB)
        df.loc[df["match_id"] == mid, "pred_home_win"] = p_home
        df.loc[df["match_id"] == mid, "pred_draw"] = p_draw
        df.loc[df["match_id"] == mid, "pred_away_win"] = p_away
        write_csv(df, POULES_PATH, GCS_POULES_BLOB, index=False, encoding="utf-8-sig")
    else:
        arbre_data = read_json(ARBRE_PATH, GCS_ARBRE_BLOB)
        for am in arbre_data:
            if int(am["match_id"]) == mid:
                am["pred_home_win"] = p_home
                am["pred_draw"] = p_draw
                am["pred_away_win"] = p_away
                break
        write_json(arbre_data, ARBRE_PATH, GCS_ARBRE_BLOB)

    # Lire les résultats existants et ajouter le nouveau
    rows = read_results_csv(REAL_RESULTS_PATH, GCS_RESULTS_BLOB)
    rows = [r for r in rows if str(r.get("match_id", "")).strip() != match_id]
    rows.append({
        "match_id": match_id,
        "home_score": hs,
        "away_score": as_,
        "result": result,
        "shootout_winner": shootout_winner
    })
    write_results_csv(rows, REAL_RESULTS_PATH, GCS_RESULTS_BLOB)

    return jsonify({"status": "ok", "result": result})

@app.route("/api/predictions")
def api_predictions():
    poules, arbre, standings, stats, current_elo, base_elo_ratings = load_predictions()
    return jsonify({"poules": poules, "arbre": arbre, "standings": standings, "stats": stats})

@app.route("/api/stats")
def api_stats():
    _, _, _, stats, _, _ = load_predictions()
    return jsonify(stats)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
