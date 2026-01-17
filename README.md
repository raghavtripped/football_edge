# Football Edge System V3 🎯⚽

A **production-grade betting edge system** for Premier League card markets. Uses multi-variable Poisson/Negative Binomial regression with proper team discipline splits, lagged referee stats, and odds-implied game-state awareness.

**Status:** ✅ **PRODUCTION READY** — Full web interface with CLV tracking.

> *"The only metric that matters now is Closing Line Value over 300-500 bets."*

---

## 🆕 What's New (Jan 2026)

### Web Interface
- **Full Flask Web UI** at `http://localhost:5001`
- **Searchable dropdowns** for teams and referees (type to filter)
- **Real-time referee stats** displayed on selection
- **Interactive bet logging** with CLV tracking

### Data Coverage
- **20 Premier League teams** (complete coverage)
- **21 referees** with dynamic stats from historical data
- **210+ matches** (Aug 2025 - Jan 2026)
- **League average: 3.84 cards/game**

### Features
- **Bet Selection** - Choose any bet to log, not just the "best"
- **"How It Works" Tab** - Step-by-step model explanation
- **Enhanced Bet History** - View all input details for past bets
- **Proper Edge Display** - Color-coded positive/negative edges

---

## Quick Start

```bash
# 1. Setup
cd football_edge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or: pip install flask pandas numpy statsmodels scipy soccerdata beautifulsoup4 lxml

# 2. Run Web Interface
python3 app.py

# 3. Open browser
open http://localhost:5001
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FOOTBALL EDGE SYSTEM V3                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐    │
│  │  FBref Data │───▶│  Feature Engine  │───▶│  Poisson/NB GLM Model   │    │
│  │ (cached)    │    │  • Split cards   │    │  • λ prediction         │    │
│  └─────────────┘    │  • Lagged refs   │    │  • Probability dist     │    │
│                     │  • Strength diff │    └───────────┬─────────────┘    │
│  ┌─────────────┐    │  • Time decay    │                │                  │
│  │  User Input │───▶│  • Derby/Top6    │                ▼                  │
│  │ (Web UI)    │    └──────────────────┘    ┌─────────────────────────┐    │
│  │ • Teams     │                            │     Edge Calculator     │    │
│  │ • Referee   │                            │  P_model - P_book = Edge│    │
│  │ • 1X2 Odds  │                            └───────────┬─────────────┘    │
│  │ • Card Odds │                                        │                  │
│  └─────────────┘                                        ▼                  │
│                                             ┌─────────────────────────┐    │
│  ┌─────────────┐                            │      Bet Logger         │    │
│  │  Flask API  │◀───────────────────────────│  • CLV tracking         │    │
│  │  (app.py)   │                            │  • Result recording     │    │
│  └─────────────┘                            │  • Win rate analysis    │    │
│        │                                    └─────────────────────────┘    │
│        ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         WEB INTERFACE                                │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────────┐                  │  │
│  │  │ 🔮 Predict│  │ 📜 Bet History│  │ 📚 How It Works│                  │  │
│  │  └──────────┘  └──────────────┘  └───────────────┘                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Web Interface Features

### 🔮 Predict Tab
- **Searchable team dropdowns** with card rate stats (H: X.X | A: X.X)
- **Searchable referee dropdown** with avg cards/game
- **1X2 odds input** for strength calculation
- **Card market odds** (1.5, 2.5, 3.5, 4.5, 5.5, 6.5 lines)
- **Real-time referee stats** on selection
- **Probability table** for all lines
- **Edge calculation** with color-coded results
- **Bet selection dropdown** - choose any bet to log

### 📜 Bet History Tab
- **CLV tracking** with closing odds
- **Win rate statistics**
- **Average edge** across all bets
- **View details** (👁️) - see all input data
- **Explain** (📚) - step-by-step model breakdown
- **Edit/Delete** functionality
- **Result recording** (Win/Loss/Push)

### 📚 How It Works Tab
- **Step-by-step model explanation**
- **Visual infographics** for each component
- **Dynamic content** based on last prediction
- **Educational glossary**

---

## Model Details

### What V3 Predicts
Predicts **total cards** in Premier League matches by modeling:

1. **Home team discipline** — Cards received by home team at home (time-decayed)
2. **Away team discipline** — Cards received by away team away (time-decayed)
3. **Referee strictness** — Computed from actual historical data (lagged, no leakage)
4. **Game-state proxy** — Strength differential from 1X2 odds
5. **Context flags** — Derby, Top 6 clash (learned coefficients)

### Key Statistics (Current Data)
```
📊 DATA SUMMARY:
   Total Matches: 210
   Date Range: Aug 15, 2025 → Jan 8, 2026
   Teams: 20 (all Premier League)
   Referees: 21
   League Avg Total Cards: 3.84
   
📊 REFEREE EXTREMES:
   Strictest: Tim Robinson (5.4 cards/game)
   Most Lenient: Craig Pawson (2.0 cards/game)
```

### Critical Fixes in V3

#### ✅ Split Team Cards
```python
# V2 (wrong): contaminated with opponent
home_cards_rate = mean(total_cards)

# V3 (correct): team's OWN cards
home_own_rate = mean(home_cards)  # at home
away_own_rate = mean(away_cards)  # away
```

#### ✅ Dynamic Referee Stats
```python
# Computed from actual data, not hardcoded
ref_stats = {
    'time_decayed_avg': 4.33,  # Recent form weighted
    'simple_avg': 3.88,
    'last_5_avg': 4.2,
    'games': 17,
    'strictness': 1.127  # vs league average
}
```

#### ✅ Odds-Implied Strength
```python
# Converts 1X2 odds to strength differential
home_strength = 1 / home_odds
away_strength = 1 / away_odds
strength_diff = home_strength - away_strength
```

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | **Flask web server** - API endpoints, model loading |
| `edge_system_v3.py` | **Core model** - Feature engineering, GLM, predictions |
| `templates/index.html` | **Web interface** - Full UI with tabs |
| `match_cache_v3.csv` | **Data cache** - 210 matches with split cards |
| `bet_log.json` | **Bet history** - CLV tracking data |
| `README.md` | This documentation |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/init` | GET | Get teams, referees, model info |
| `/api/predict` | POST | Run prediction with input data |
| `/api/referee/<name>` | GET | Get detailed referee stats |
| `/api/team/<name>` | GET | Get team card rates |
| `/api/log_bet` | POST | Log a bet for CLV tracking |
| `/api/bet_log` | GET | Get all logged bets |
| `/api/bet_stats` | GET | Get aggregate statistics |
| `/api/update_bet/<id>` | PUT | Update closing odds/result |
| `/api/delete_bet/<id>` | DELETE | Remove a logged bet |

---

## Current Team & Referee Coverage

### Teams (20)
Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Chelsea, Crystal Palace, Everton, Fulham, Leeds United, Liverpool, Manchester City, Manchester Utd, Newcastle Utd, Nott'ham Forest, Sunderland, Tottenham, West Ham, Wolves

### Referees (21) with Avg Cards/Game
| Referee | Games | Avg Cards |
|---------|-------|-----------|
| Tim Robinson | 5 | 5.4 ⬆️ |
| Stuart Attwell | 14 | 4.9 |
| Michael Salisbury | 8 | 4.9 |
| Peter Bankes | 15 | 4.6 |
| Simon Hooper | 12 | 4.4 |
| John Brooks | 7 | 4.3 |
| Chris Kavanagh | 16 | 4.1 |
| Samuel Barrott | 12 | 4.1 |
| Darren England | 13 | 4.0 |
| Anthony Taylor | 17 | 3.9 |
| Andrew Kitchen | 3 | 3.7 |
| Thomas Bramall | 12 | 3.6 |
| Robert Jones | 12 | 3.6 |
| Andy Madley | 10 | 3.5 |
| Jarred Gillett | 11 | 3.5 |
| Tony Harrington | 10 | 3.1 |
| Michael Oliver | 16 | 2.8 |
| Craig Pawson | 12 | 2.0 ⬇️ |

---

## Validation: CLV is Everything

**Nothing in the code matters except CLV.**

```
If CLV > 0 consistently → Edge exists
If CLV ≤ 0           → Model is wrong
```

### How to Validate:
1. Log all bets via web interface
2. Record **closing odds** before kickoff
3. Mark result after match (Win/Loss/Push)
4. Track CLV: `CLV = (closing_odds / opening_odds) - 1`
5. **Positive CLV over 300+ bets = real edge**

### Sample Size Requirements
- **Minimum:** 300 bets
- **Preferred:** 500+ bets
- **High confidence:** 1000+ bets

---

## Installation

```bash
# Clone/download the project
cd football_edge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install flask pandas numpy statsmodels scipy soccerdata beautifulsoup4 lxml

# Run the web interface
python3 app.py

# Open in browser
open http://localhost:5001
```

---

## Usage Tips

### Making a Prediction
1. Select **Home Team** and **Away Team**
2. Select **Referee** (optional but recommended)
3. Enter **1X2 odds** for strength calculation
4. Enter **card market odds** for lines you want to analyze
5. Click **Run Prediction**
6. Review probabilities and edges
7. **Log bet** for CLV tracking

### Logging Non-Value Bets
Even if no value bet is found (edge < 5%), you can:
1. Use the **bet selector dropdown** to choose any bet
2. Click **"Log for CLV"** to track it anyway
3. Useful for analyzing marginal bets

### Reviewing Historical Bets
1. Go to **Bet History** tab
2. Click **👁️** to view input details
3. Click **📚** to see model explanation
4. Click **Edit** to add closing odds/result
5. Monitor **Avg CLV** over time

---

## What This System IS vs IS NOT

### ✅ What it IS
- Pre-match football card totals pricing engine
- Referee bias modeling with actual data
- Time-varying team discipline
- Odds-implied game state
- Properly distributed probabilities (Poisson/NB)
- Auditable CLV pipeline
- Full web interface for ease of use

### ❌ What it is NOT
- Live betting system
- Team totals engine (home vs away separate)
- Guaranteed profit generator
- A theoretical exercise — **it proves itself empirically**

---

## Changelog

### v3.2 (Jan 17, 2026)
- ✅ Full web interface with Flask
- ✅ Searchable dropdowns for teams/referees
- ✅ Dynamic referee stats from historical data
- ✅ Bet selection dropdown (choose any bet)
- ✅ "How It Works" educational tab
- ✅ Enhanced bet history with view/explain
- ✅ Proper edge color coding (positive/negative)
- ✅ All 20 PL teams + 21 referees

### v3.1 (Jan 16, 2026)
- ✅ Expanded data coverage (210 matches)
- ✅ Removed hardcoded referee overrides
- ✅ API for dynamic referee stats

### v3.0 (Initial)
- ✅ Split team cards (fixed V2 contamination)
- ✅ Lagged referee stats (no leakage)
- ✅ Odds-implied strength differential
- ✅ Time-decayed rates
- ✅ CLV tracking pipeline

---

*Built for finding edges in booking markets. Validate with CLV. Bet responsibly.* 🎲
