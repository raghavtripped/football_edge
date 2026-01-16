"""
Football Edge System - Web Interface
Run locally with: python3 app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from scipy.stats import nbinom as scipy_nbinom
import soccerdata as sd
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- CONFIGURATION ---
LEAGUE = "ENG-Premier League"
SEASON = "2526"
CACHE_FILE = "match_cache_v3.csv"
BET_LOG_FILE = "bet_log.json"
TIME_DECAY_HALF_LIFE = 30
OVERDISPERSION_THRESHOLD = 1.25

# --- REFEREE FALLBACK ---
REFEREE_FALLBACK = {
    'Tim Robinson': 1.35, 'Michael Salisbury': 1.27, 'Stuart Attwell': 1.21,
    'John Brooks': 1.15, 'Peter Bankes': 1.11, 'Simon Hooper': 1.10,
    'Andy Madley': 1.08, 'Samuel Barrott': 1.04, 'Chris Kavanagh': 1.01,
    'Anthony Taylor': 0.99, 'Darren England': 0.98, 'Paul Tierney': 0.96,
    'Robert Jones': 0.93, 'Jarred Gillett': 0.87, 'Tony Harrington': 0.78,
    'Michael Oliver': 0.66, 'Craig Pawson': 0.50
}

MAJOR_DERBIES = [
    ("Manchester Utd", "Manchester City"),
    ("Arsenal", "Tottenham"), ("Arsenal", "Tottenham Hotspur"),
    ("Liverpool", "Everton"),
    ("Liverpool", "Manchester Utd"),
]

TOP_6 = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester Utd",
         "Tottenham", "Tottenham Hotspur"]

# --- TEAM NAME NORMALIZATION (Issue 2 fix) ---
TEAM_ALIASES = {
    # Common variations
    "Man United": "Manchester Utd",
    "Man Utd": "Manchester Utd",
    "Manchester United": "Manchester Utd",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove": "Brighton and Hove Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "West Ham": "West Ham United",
    "West Ham Utd": "West Ham United",
    "Leicester": "Leicester City",
    "Ipswich": "Ipswich Town",
}

def normalize_team_name(name):
    """Normalize team name to canonical form."""
    if not name:
        return name
    # Check aliases first
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    # Also check with stripped whitespace
    stripped = name.strip()
    if stripped in TEAM_ALIASES:
        return TEAM_ALIASES[stripped]
    return stripped

# Global model bundle (loaded once)
MODEL_BUNDLE = None


def nb_cdf(k, mu, alpha):
    """Proper Negative Binomial CDF."""
    if alpha <= 0:
        return poisson.cdf(k, mu)
    n = 1 / alpha
    p = 1 / (1 + alpha * mu)
    return scipy_nbinom.cdf(k, n, p)


def odds_to_implied_strength(odds_dict):
    """Convert 1X2 odds to implied strength differential."""
    if not odds_dict or 'home' not in odds_dict:
        return None
    
    p_home = 1 / float(odds_dict.get('home', 2.5))
    p_draw = 1 / float(odds_dict.get('draw', 3.5))
    p_away = 1 / float(odds_dict.get('away', 2.5))
    
    total = p_home + p_draw + p_away
    p_home /= total
    p_away /= total
    
    return (p_home - p_away) * 3


def compute_time_decayed_rate(values, dates, reference_date, half_life_days=TIME_DECAY_HALF_LIFE):
    """Compute weighted average with exponential time decay."""
    if len(values) == 0:
        return None
    
    days_ago = (reference_date - dates).dt.days
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    
    weighted_sum = np.sum(values * weights)
    total_weight = np.sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else np.mean(values)


def load_and_train_model():
    """Load data and train the model."""
    global MODEL_BUNDLE
    
    if not os.path.exists(CACHE_FILE):
        return None, "Cache file not found. Run edge_system_v3.py first to generate data."
    
    df = pd.read_csv(CACHE_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # Compute team stats with time decay
    reference_date = df['date'].max()
    
    home_rates_dict = {}
    away_rates_dict = {}
    
    for team in df['home_team'].unique():
        team_home = df[df['home_team'] == team]
        if len(team_home) > 0:
            decayed_avg = compute_time_decayed_rate(
                team_home['home_cards'].values,
                team_home['date'],
                reference_date
            )
            home_rates_dict[team] = decayed_avg
    
    for team in df['away_team'].unique():
        team_away = df[df['away_team'] == team]
        if len(team_away) > 0:
            decayed_avg = compute_time_decayed_rate(
                team_away['away_cards'].values,
                team_away['date'],
                reference_date
            )
            away_rates_dict[team] = decayed_avg
    
    # Compute lagged referee stats
    df_sorted = df.sort_values('date').reset_index(drop=True)
    ref_cumulative = {}
    lagged_strictness = []
    league_cumsum = 0
    league_count = 0
    league_avg = df['total_cards'].mean()
    
    for idx, row in df_sorted.iterrows():
        ref = row['referee']
        
        if pd.isna(ref):
            lagged_strictness.append(1.0)
        elif ref in ref_cumulative and ref_cumulative[ref]['count'] >= 2:
            ref_avg = ref_cumulative[ref]['sum'] / ref_cumulative[ref]['count']
            l_avg = league_cumsum / league_count if league_count > 0 else league_avg
            lagged_strictness.append(ref_avg / l_avg if l_avg > 0 else 1.0)
        else:
            lagged_strictness.append(REFEREE_FALLBACK.get(ref, 1.0))
        
        if pd.notna(ref):
            if ref not in ref_cumulative:
                ref_cumulative[ref] = {'sum': 0, 'count': 0}
            ref_cumulative[ref]['sum'] += row['total_cards']
            ref_cumulative[ref]['count'] += 1
        
        league_cumsum += row['total_cards']
        league_count += 1
    
    df_sorted['ref_strictness_lagged'] = lagged_strictness
    
    # Compute strength
    home_perf = df.groupby('home_team').agg({'home_goals': 'mean', 'away_goals': 'mean'})
    home_perf['home_gd'] = home_perf['home_goals'] - home_perf['away_goals']
    
    away_perf = df.groupby('away_team').agg({'away_goals': 'mean', 'home_goals': 'mean'})
    away_perf['away_gd'] = away_perf['away_goals'] - away_perf['home_goals']
    
    strength = {}
    for team in set(home_perf.index) | set(away_perf.index):
        h_gd = home_perf.loc[team, 'home_gd'] if team in home_perf.index else 0
        a_gd = away_perf.loc[team, 'away_gd'] if team in away_perf.index else 0
        strength[team] = (h_gd + a_gd) / 2
    
    # Map features
    df_sorted['home_own_rate'] = df_sorted['home_team'].map(home_rates_dict).fillna(df['home_cards'].mean())
    df_sorted['away_own_rate'] = df_sorted['away_team'].map(away_rates_dict).fillna(df['away_cards'].mean())
    
    def get_strength_diff(row):
        h = strength.get(row['home_team'], 0)
        a = strength.get(row['away_team'], 0)
        return h - a
    
    df_sorted['strength_diff'] = df_sorted.apply(get_strength_diff, axis=1)
    df_sorted['away_is_underdog'] = (df_sorted['strength_diff'] > 0.5).astype(int)
    
    def is_major_derby(row):
        pair = (row['home_team'], row['away_team'])
        rev = (row['away_team'], row['home_team'])
        return 1 if pair in MAJOR_DERBIES or rev in MAJOR_DERBIES else 0
    
    def is_top6(row):
        return 1 if row['home_team'] in TOP_6 and row['away_team'] in TOP_6 else 0
    
    df_sorted['is_major_derby'] = df_sorted.apply(is_major_derby, axis=1)
    df_sorted['is_top6_clash'] = df_sorted.apply(is_top6, axis=1)
    
    # Time weights
    days_ago = (reference_date - df_sorted['date']).dt.days
    weights = np.exp(-np.log(2) * days_ago / TIME_DECAY_HALF_LIFE)
    
    # Train model
    feature_cols = ['home_own_rate', 'away_own_rate', 'ref_strictness_lagged', 
                    'strength_diff', 'away_is_underdog', 'is_major_derby', 'is_top6_clash']
    
    X = df_sorted[feature_cols].copy()
    X = sm.add_constant(X)
    y = df_sorted['total_cards']
    
    # Check dispersion
    dispersion = np.var(y) / np.mean(y)
    
    if dispersion > OVERDISPERSION_THRESHOLD:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), freq_weights=weights).fit()
        model_type = "NegBin"
        try:
            alpha = model.scale
        except:
            alpha = dispersion - 1
    else:
        model = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=weights).fit()
        model_type = "Poisson"
        alpha = 0
    
    MODEL_BUNDLE = {
        'model': model,
        'model_type': model_type,
        'alpha': alpha,
        'home_rates': home_rates_dict,
        'away_rates': away_rates_dict,
        'ref_stats': ref_cumulative,
        'strength': strength,
        'league_avg': league_avg,
        'home_cards_avg': df['home_cards'].mean(),
        'away_cards_avg': df['away_cards'].mean(),
        'dispersion': dispersion,
        'teams': sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique()))),
        'referees': sorted(list(REFEREE_FALLBACK.keys())),
    }
    
    return MODEL_BUNDLE, None


def predict_single_match(home, away, referee, odds):
    """Predict a single match."""
    if MODEL_BUNDLE is None:
        return None, "Model not loaded"
    
    model = MODEL_BUNDLE['model']
    model_type = MODEL_BUNDLE['model_type']
    alpha = MODEL_BUNDLE['alpha']
    home_rates = MODEL_BUNDLE['home_rates']
    away_rates = MODEL_BUNDLE['away_rates']
    ref_stats = MODEL_BUNDLE['ref_stats']
    strength = MODEL_BUNDLE['strength']
    league_avg = MODEL_BUNDLE['league_avg']
    
    # Team rates
    home_rate = home_rates.get(home, MODEL_BUNDLE['home_cards_avg'])
    away_rate = away_rates.get(away, MODEL_BUNDLE['away_cards_avg'])
    
    # Referee (Issue 1 fix: use ONLY fallback to prevent leakage)
    # ref_stats contains future information relative to prediction time
    # Safe: use precomputed fallback strictness only
    if referee:
        ref_strict = REFEREE_FALLBACK.get(referee, 1.0)
    else:
        ref_strict = 1.0
    
    # Strength (prefer odds-implied)
    odds_strength = odds_to_implied_strength(odds) if odds else None
    using_odds_strength = odds_strength is not None
    
    if using_odds_strength:
        str_diff = odds_strength
    else:
        h_str = strength.get(home, 0)
        a_str = strength.get(away, 0)
        str_diff = h_str - a_str
    
    # Binary features
    pair = (home, away)
    rev = (away, home)
    is_derby = 1 if pair in MAJOR_DERBIES or rev in MAJOR_DERBIES else 0
    is_top6 = 1 if home in TOP_6 and away in TOP_6 else 0
    
    # Issue 3 fix: Disable is_underdog when using odds-implied strength
    # to prevent collinearity (odds already encode dominance)
    if using_odds_strength:
        is_underdog = 0
    else:
        is_underdog = 1 if str_diff > 0.5 else 0
    
    # Predict
    X_pred = pd.DataFrame({
        'const': [1],
        'home_own_rate': [home_rate],
        'away_own_rate': [away_rate],
        'ref_strictness_lagged': [ref_strict],
        'strength_diff': [str_diff],
        'away_is_underdog': [is_underdog],
        'is_major_derby': [is_derby],
        'is_top6_clash': [is_top6],
    })
    
    lambda_val = model.predict(X_pred)[0]
    
    # Probabilities - All standard lines (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)
    if model_type == "Poisson":
        probs = {
            'o15': 1 - poisson.cdf(1, lambda_val),
            'o25': 1 - poisson.cdf(2, lambda_val),
            'o35': 1 - poisson.cdf(3, lambda_val),
            'o45': 1 - poisson.cdf(4, lambda_val),
            'o55': 1 - poisson.cdf(5, lambda_val),
            'o65': 1 - poisson.cdf(6, lambda_val),
            'u15': poisson.cdf(1, lambda_val),
            'u25': poisson.cdf(2, lambda_val),
            'u35': poisson.cdf(3, lambda_val),
            'u45': poisson.cdf(4, lambda_val),
            'u55': poisson.cdf(5, lambda_val),
            'u65': poisson.cdf(6, lambda_val),
        }
    else:
        probs = {
            'o15': 1 - nb_cdf(1, lambda_val, alpha),
            'o25': 1 - nb_cdf(2, lambda_val, alpha),
            'o35': 1 - nb_cdf(3, lambda_val, alpha),
            'o45': 1 - nb_cdf(4, lambda_val, alpha),
            'o55': 1 - nb_cdf(5, lambda_val, alpha),
            'o65': 1 - nb_cdf(6, lambda_val, alpha),
            'u15': nb_cdf(1, lambda_val, alpha),
            'u25': nb_cdf(2, lambda_val, alpha),
            'u35': nb_cdf(3, lambda_val, alpha),
            'u45': nb_cdf(4, lambda_val, alpha),
            'u55': nb_cdf(5, lambda_val, alpha),
            'u65': nb_cdf(6, lambda_val, alpha),
        }
    
    # Fair odds
    fair = {k: round(1/v, 2) if v > 0.01 else 99 for k, v in probs.items()}
    
    # Edge calculation - check all lines that have odds provided
    edges = {}
    all_lines = ['o15', 'u15', 'o25', 'u25', 'o35', 'u35', 'o45', 'u45', 'o55', 'u55', 'o65', 'u65']
    if odds:
        for line in all_lines:
            if line in odds:
                book_implied = 1 / float(odds[line])
                model_prob = probs.get(line, 0)
                edges[line] = round((model_prob - book_implied) * 100, 1)
    
    # Best bet
    best_bet = None
    best_edge = 0
    for line, edge in edges.items():
        if edge > best_edge:
            best_edge = edge
            best_bet = line
    
    return {
        'lambda': round(lambda_val, 2),
        'probs': {k: round(v * 100, 1) for k, v in probs.items()},
        'fair_odds': fair,
        'edges': edges,
        'best_bet': best_bet,
        'best_edge': best_edge,
        'model_type': model_type,
        'features': {
            'home_rate': round(home_rate, 2),
            'away_rate': round(away_rate, 2),
            'ref_strict': round(ref_strict, 2),
            'strength_diff': round(str_diff, 2),
            'is_derby': is_derby,
            'is_top6': is_top6,
        }
    }, None


def log_bet(match, line, model_prob, book_odds, edge, lambda_val):
    """Log a bet for CLV tracking.
    
    Caveat 1 fix: model_prob stored as raw probability (0-1), not percentage.
    If frontend sends percentage, convert it here.
    """
    # Ensure raw probability (0-1) not percentage
    prob_value = float(model_prob) if model_prob else 0
    if prob_value > 1:
        prob_value = prob_value / 100  # Convert percentage to probability
    
    bet = {
        'timestamp': datetime.now().isoformat(),
        'match': match,
        'line': line,
        'model_prob_raw': round(prob_value, 4),  # Store as 0-1
        'model_prob_pct': round(prob_value * 100, 1),  # Also store for display
        'book_odds': book_odds,
        'edge': edge,
        'lambda': lambda_val,
        'closing_odds': None,
        'result': None,
    }
    
    if os.path.exists(BET_LOG_FILE):
        with open(BET_LOG_FILE, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    log.append(bet)
    
    with open(BET_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)
    
    return bet


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/init', methods=['GET'])
def api_init():
    """Initialize model and return teams/referees."""
    bundle, error = load_and_train_model()
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'teams': bundle['teams'],
        'referees': bundle['referees'],
        'model_type': bundle['model_type'],
        'dispersion': round(bundle['dispersion'], 2),
        'league_avg': round(bundle['league_avg'], 2),
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict a single match."""
    data = request.json
    
    home = data.get('home')
    away = data.get('away')
    referee = data.get('referee')
    odds = data.get('odds', {})
    
    if not home or not away:
        return jsonify({'error': 'Home and away teams required'}), 400
    
    # Issue 2 fix: Normalize team names
    home = normalize_team_name(home)
    away = normalize_team_name(away)
    
    # Caveat 2 fix: Validate odds (minimum sanity check)
    for key, val in list(odds.items()):
        try:
            odds_val = float(val)
            if odds_val <= 1.01 or odds_val > 100:
                return jsonify({'error': f'Invalid odds for {key}: must be > 1.01 and < 100'}), 400
        except (ValueError, TypeError):
            if val:  # Only error if non-empty
                return jsonify({'error': f'Invalid odds format for {key}'}), 400
    
    result, error = predict_single_match(home, away, referee, odds)
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify(result)


@app.route('/api/log_bet', methods=['POST'])
def api_log_bet():
    """Log a bet for CLV tracking."""
    data = request.json
    
    bet = log_bet(
        match=data.get('match'),
        line=data.get('line'),
        model_prob=data.get('model_prob'),
        book_odds=data.get('book_odds'),
        edge=data.get('edge'),
        lambda_val=data.get('lambda'),
    )
    
    return jsonify({'success': True, 'bet': bet})


@app.route('/api/bet_log', methods=['GET'])
def api_bet_log():
    """Get all logged bets."""
    if os.path.exists(BET_LOG_FILE):
        with open(BET_LOG_FILE, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    # Add index to each bet for identification
    for i, bet in enumerate(log):
        bet['id'] = i
    
    return jsonify(log)


@app.route('/api/bet_log/<int:bet_id>', methods=['PUT'])
def api_update_bet(bet_id):
    """Update a specific bet."""
    if not os.path.exists(BET_LOG_FILE):
        return jsonify({'error': 'No bets found'}), 404
    
    with open(BET_LOG_FILE, 'r') as f:
        log = json.load(f)
    
    if bet_id < 0 or bet_id >= len(log):
        return jsonify({'error': 'Bet not found'}), 404
    
    data = request.json
    
    # Update allowed fields
    if 'closing_odds' in data:
        log[bet_id]['closing_odds'] = data['closing_odds']
    if 'result' in data:
        log[bet_id]['result'] = data['result']
    if 'notes' in data:
        log[bet_id]['notes'] = data['notes']
    
    # Calculate CLV if we have closing odds
    if log[bet_id].get('closing_odds') and log[bet_id].get('book_odds'):
        book = float(log[bet_id]['book_odds'])
        closing = float(log[bet_id]['closing_odds'])
        # CLV = (closing implied prob - book implied prob) * 100
        # Positive CLV means you got better odds than closing
        clv = ((1/closing) - (1/book)) * 100
        log[bet_id]['clv'] = round(clv, 2)
    
    with open(BET_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)
    
    log[bet_id]['id'] = bet_id
    return jsonify({'success': True, 'bet': log[bet_id]})


@app.route('/api/bet_log/<int:bet_id>', methods=['DELETE'])
def api_delete_bet(bet_id):
    """Delete a specific bet."""
    if not os.path.exists(BET_LOG_FILE):
        return jsonify({'error': 'No bets found'}), 404
    
    with open(BET_LOG_FILE, 'r') as f:
        log = json.load(f)
    
    if bet_id < 0 or bet_id >= len(log):
        return jsonify({'error': 'Bet not found'}), 404
    
    deleted = log.pop(bet_id)
    
    with open(BET_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)
    
    return jsonify({'success': True, 'deleted': deleted})


@app.route('/api/bet_stats', methods=['GET'])
def api_bet_stats():
    """Get aggregate statistics for all bets."""
    if not os.path.exists(BET_LOG_FILE):
        return jsonify({
            'total_bets': 0,
            'pending': 0,
            'won': 0,
            'lost': 0,
            'win_rate': None,
            'avg_edge': None,
            'avg_clv': None,
        })
    
    with open(BET_LOG_FILE, 'r') as f:
        log = json.load(f)
    
    total = len(log)
    won = sum(1 for b in log if b.get('result') == 'won')
    lost = sum(1 for b in log if b.get('result') == 'lost')
    pending = total - won - lost
    
    edges = [b.get('edge', 0) for b in log if b.get('edge') is not None]
    clvs = [b.get('clv', 0) for b in log if b.get('clv') is not None]
    
    return jsonify({
        'total_bets': total,
        'pending': pending,
        'won': won,
        'lost': lost,
        'win_rate': round(won / (won + lost) * 100, 1) if (won + lost) > 0 else None,
        'avg_edge': round(sum(edges) / len(edges), 1) if edges else None,
        'avg_clv': round(sum(clvs) / len(clvs), 2) if clvs else None,
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 FOOTBALL EDGE SYSTEM - Web Interface")
    print("="*60)
    print("\n   Loading model...")
    
    bundle, error = load_and_train_model()
    if error:
        print(f"   ❌ Error: {error}")
    else:
        print(f"   ✅ Model loaded ({bundle['model_type']})")
        print(f"   📊 {len(bundle['teams'])} teams, {len(bundle['referees'])} referees")
    
    print("\n   Starting server...")
    print("   Open: http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001)
