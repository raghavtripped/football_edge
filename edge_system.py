import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
import soccerdata as sd
import os
import warnings
from datetime import datetime

# --- CONFIGURATION ---
LEAGUE = "ENG-Premier League"
SEASON = "2526"  # "2526" covers the 2025/2026 season (current season as of Jan 2026)
CACHE_FILE = "match_cache.csv"

# --- MANUAL REFEREE OVERRIDES (when FBref lags behind PGMOL) ---
# Format: "Home vs Away": "Referee Name" (use exact FBref team names)
MANUAL_REF_OVERRIDES = {
    "Manchester Utd vs Manchester City": "Anthony Taylor",
    "Nottingham Forest vs Arsenal": "Michael Oliver",
    "Nott'ham Forest vs Arsenal": "Michael Oliver",  # alternate spelling
    "Tottenham Hotspur vs West Ham United": "Jarred Gillett",
    "Tottenham vs West Ham": "Jarred Gillett",  # alternate
    "Wolverhampton Wanderers vs Newcastle United": "Samuel Barrott",
    "Wolves vs Newcastle Utd": "Samuel Barrott",  # alternate
    "Wolves vs Newcastle United": "Samuel Barrott",  # alternate
    "Chelsea vs Brentford": "John Brooks",
    "Sunderland vs Crystal Palace": "Robert Jones",
    "Brighton vs Bournemouth": "Andy Madley",
    "Brighton and Hove Albion vs Bournemouth": "Andy Madley",  # alternate
    "Aston Villa vs Everton": "Peter Bankes",
    "Leeds United vs Fulham": "Darren England",
    "Liverpool vs Burnley": "Stuart Attwell",
}

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. THE REFEREE KNOWLEDGE BASE (Human Inputs) ---
# Updated for 2024/25 trends
REFEREE_STRICTNESS = {
    'Tim Robinson': 1.35,
    'Michael Salisbury': 1.27,
    'Stuart Attwell': 1.21,
    'John Brooks': 1.15,       # Strict-ish
    'Peter Bankes': 1.11,
    'Simon Hooper': 1.10,
    'Andy Madley': 1.08,       # Avg+
    'Samuel Barrott': 1.04,
    'Chris Kavanagh': 1.01,
    'Anthony Taylor': 0.99,
    'Darren England': 0.98,
    'Paul Tierney': 0.96,      # Avg-
    'Robert Jones': 0.93,
    'Jarred Gillett': 0.87,
    'Tony Harrington': 0.78,
    'Michael Oliver': 0.66,    # The Trap
    'Craig Pawson': 0.50       # The Widowmaker
}

def get_ref_score(ref_name):
    if pd.isna(ref_name): return 1.0
    return REFEREE_STRICTNESS.get(ref_name, 1.0)

# --- 2. THE REAL DATA ENGINE (With Caching) ---
def get_training_data():
    """
    Checks for local CSV. If missing, scrapes FBref for ~15 mins.
    Returns a clean DataFrame with 'total_cards' for past matches.
    """
    
    # A. TRY LOADING CACHE
    if os.path.exists(CACHE_FILE):
        print(f"📂 Found local cache ({CACHE_FILE}). Loading instantly...")
        return pd.read_csv(CACHE_FILE)

    # B. NO CACHE? SCRAPE IT.
    print(f"⚠️ No cache found. Initiating full scrape for {LEAGUE} {SEASON}...")
    print("☕ This will take 10-20 minutes. Go grab a coffee.")
    print("   > Connecting to FBref...")
    
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASON)
    
    # We need "misc" stats because that's where cards (CrdY, CrdR) live in logs
    print("   > Downloading Match Logs (stat_type='misc')...")
    # This returns a multi-index DF: (League, Season, Team, Game)
    logs = fbref.read_team_match_stats(stat_type="misc")
    
    # Also get schedule for referee info
    print("   > Downloading Schedule (for referee data)...")
    schedule = fbref.read_schedule()
    
    print("   > Processing Data...")
    
    # Reset index to make 'game' (match ID/date) accessible
    logs = logs.reset_index()
    
    # Flatten multi-level columns: ('Performance', 'CrdY') -> 'Performance_CrdY'
    logs.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in logs.columns]
    
    # Extract total cards (Yellow + Red)
    # Column names after flattening: 'Performance_CrdY', 'Performance_CrdR'
    logs['team_cards'] = logs['Performance_CrdY'] + logs['Performance_CrdR']
    
    processed_rows = []
    
    # Create a unique match ID for every row to enable grouping
    def make_match_id(row):
        teams = sorted([str(row['team']), str(row['opponent'])])
        date_str = str(row['date']).split()[0]  # Get just the date part
        return f"{date_str}_{teams[0]}_{teams[1]}"

    logs['match_id'] = logs.apply(make_match_id, axis=1)
    
    # Group by this unique ID to combine Home and Away stats
    for match_id, group in logs.groupby('match_id'):
        if len(group) < 2: continue # partial data
        
        # Determine Home/Away (FBref has 'venue' column)
        home_rows = group[group['venue'] == 'Home']
        away_rows = group[group['venue'] == 'Away']
        
        if len(home_rows) == 0 or len(away_rows) == 0:
            continue
            
        home_row = home_rows.iloc[0]
        away_row = away_rows.iloc[0]
        
        total_match_cards = home_row['team_cards'] + away_row['team_cards']
        
        processed_rows.append({
            'date': str(home_row['date']).split()[0],
            'home': home_row['team'],
            'away': away_row['team'],
            'total_cards': total_match_cards
        })
        
    final_df = pd.DataFrame(processed_rows)
    
    # Merge with schedule to get referee info
    schedule = schedule.reset_index()
    schedule['date_str'] = schedule['date'].astype(str).str.split().str[0]
    
    # Create lookup dict for referees
    ref_lookup = {}
    for _, row in schedule.iterrows():
        key = f"{row['date_str']}_{row['home_team']}_{row['away_team']}"
        ref_lookup[key] = row.get('referee', None)
    
    # Add referee to final_df
    def get_referee(row):
        key = f"{row['date']}_{row['home']}_{row['away']}"
        return ref_lookup.get(key, None)
    
    final_df['referee'] = final_df.apply(get_referee, axis=1)
    
    # Save to CSV so we never have to wait again
    final_df.to_csv(CACHE_FILE, index=False)
    print(f"✅ Data scraped & saved to {CACHE_FILE}. ({len(final_df)} matches)")
    
    return final_df

# --- 3. THE ENHANCED MODEL & PREDICTION ---
def run_system():
    # 1. Get History (Real Data)
    history_df = get_training_data()
    
    print("   > Computing Team Card Tendencies...")
    
    # 2. Compute TEAM-LEVEL card tendencies from actual data
    # Home team average cards in home games
    home_card_avg = history_df.groupby('home')['total_cards'].mean().to_dict()
    # Away team average cards in away games  
    away_card_avg = history_df.groupby('away')['total_cards'].mean().to_dict()
    
    # League average for fallback
    league_avg = history_df['total_cards'].mean()
    
    # Compute referee averages from ACTUAL data (not just the dict)
    ref_card_avg = history_df.groupby('referee')['total_cards'].mean().to_dict()
    
    # 3. Add Features to training data
    history_df['home_tendency'] = history_df['home'].map(home_card_avg)
    history_df['away_tendency'] = history_df['away'].map(away_card_avg)
    history_df['ref_avg'] = history_df['referee'].map(ref_card_avg)
    history_df['ref_strictness'] = history_df['referee'].apply(get_ref_score)
    
    # 4. Train MULTI-VARIABLE Poisson Model
    print("   > Training Enhanced Poisson Model...")
    print("     Features: Home Tendency + Away Tendency + Ref Strictness")
    
    X = history_df[['home_tendency', 'away_tendency', 'ref_strictness']]
    X = sm.add_constant(X)
    y = history_df['total_cards']
    
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    print(f"\n   📊 MODEL COEFFICIENTS:")
    print(f"      Intercept:      {model.params['const']:.3f}")
    print(f"      Home Tendency:  {model.params['home_tendency']:.3f}")
    print(f"      Away Tendency:  {model.params['away_tendency']:.3f}")
    print(f"      Ref Strictness: {model.params['ref_strictness']:.3f}")
    print(f"\n   📈 League Avg Cards: {league_avg:.2f}")

    # 5. Predict Next Matches (Live Schedule)
    print("\n⚡ PULLING UPCOMING FIXTURES...")
    fbref_live = sd.FBref(leagues=LEAGUE, seasons=SEASON)
    schedule = fbref_live.read_schedule()
    
    # Filter for future games (no score yet)
    upcoming = schedule[schedule['score'].isna()].head(10)
    
    print("\n" + "="*85)
    print(f"{'MATCH':<35} | {'REF':<16} | {'H_AVG':>5} | {'A_AVG':>5} | {'EXP':>5} | {'SIGNAL'}")
    print("-" * 85)

    for idx, row in upcoming.iterrows():
        match_key = f"{row['home_team']} vs {row['away_team']}"
        
        # Check for manual referee override first
        if match_key in MANUAL_REF_OVERRIDES:
            ref_name = MANUAL_REF_OVERRIDES[match_key]
            ref_source = "📋"
        elif pd.notna(row['referee']):
            ref_name = row['referee']
            ref_source = ""
        else:
            ref_name = None
            ref_source = ""
        
        # Get all features
        home_tend = home_card_avg.get(row['home_team'], league_avg)
        away_tend = away_card_avg.get(row['away_team'], league_avg)
        ref_score = get_ref_score(ref_name)
        
        # Predict using full model
        pred_input = pd.DataFrame({
            'const': [1], 
            'home_tendency': [home_tend],
            'away_tendency': [away_tend],
            'ref_strictness': [ref_score]
        })
        lambda_val = model.predict(pred_input)[0]
        
        # Derby adjustments (check for partial team name matches)
        is_derby = 0
        derby_tag = ""
        home = row['home_team'].lower()
        away = row['away_team'].lower()
        
        # Manchester Derby
        if ('manchester' in home or 'man utd' in home or 'man city' in home) and \
           ('manchester' in away or 'man utd' in away or 'man city' in away) and home != away:
            is_derby = 0.7
            derby_tag = "🔥"
        # North London Derby
        elif ('tottenham' in home or 'spurs' in home) and 'arsenal' in away:
            is_derby = 0.7
            derby_tag = "🔥"
        elif 'arsenal' in home and ('tottenham' in away or 'spurs' in away):
            is_derby = 0.7
            derby_tag = "🔥"
        # Merseyside Derby / Liverpool rivalries
        elif 'liverpool' in home and ('everton' in away or 'manchester' in away):
            is_derby = 0.5
            derby_tag = "🔥"
        elif 'liverpool' in away and ('everton' in home or 'manchester' in home):
            is_derby = 0.5
            derby_tag = "🔥"
        # London Derbies
        elif any(x in home for x in ['tottenham', 'west ham', 'chelsea', 'arsenal', 'crystal palace', 'fulham', 'brentford']) and \
             any(x in away for x in ['tottenham', 'west ham', 'chelsea', 'arsenal', 'crystal palace', 'fulham', 'brentford']):
            is_derby = 0.4
            derby_tag = "⚔️"
            
        final_exp = lambda_val + is_derby
        
        # Odds Calculations
        prob_o25 = 1 - poisson.cdf(2, final_exp)
        prob_o35 = 1 - poisson.cdf(3, final_exp)
        prob_o45 = 1 - poisson.cdf(4, final_exp)
        prob_o55 = 1 - poisson.cdf(5, final_exp)
        
        fair_35 = 1/prob_o35 if prob_o35 > 0.01 else 99
        fair_45 = 1/prob_o45 if prob_o45 > 0.01 else 99
        fair_55 = 1/prob_o55 if prob_o55 > 0.01 else 99
        
        # Generate signal based on expected value and probabilities
        signal = "-"
        if final_exp > 5.5:
            signal = f"🔥🔥 O5.5 (@{fair_55:.2f})"
        elif final_exp > 4.8:
            signal = f"🔥 O4.5 (@{fair_45:.2f})"
        elif final_exp > 4.2:
            signal = f"✅ O4.5 (@{fair_45:.2f})"
        elif final_exp > 3.8:
            signal = f"📈 O3.5 (@{fair_35:.2f})"
        elif final_exp < 3.0:
            signal = f"⚠️ UNDER 3.5"
        elif ref_score < 0.75:
            signal = "⚠️ U4.5?"
        
        ref_display = f"{ref_source}{ref_name}" if ref_name else "<NA>"

        print(f"{derby_tag}{row['home_team']:<16} vs {row['away_team']:<15} | {ref_display:<16} | {home_tend:>5.2f} | {away_tend:>5.2f} | {final_exp:>5.2f} | {signal}")

if __name__ == "__main__":
    run_system()
