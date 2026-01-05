# Dataset: Corner Kick Video and Spatial Data

This document describes all data sources used for corner kick prediction.

---

## 1. Overview

Two data sources were used:

| Source | Type | Corners | Use |
|--------|------|---------|-----|
| SoccerNet | Broadcast video | 4,836 | FAANTRA model |
| StatsBomb | 360° freeze frames | 1,933 | Classical ML baseline |

---

## 2. SoccerNet Video Dataset

### 2.1 Source

**SoccerNet**: Large-scale soccer video dataset
- Website: https://www.soccer-net.org/
- 550 full matches from 6 European leagues
- 720p broadcast quality video
- 25 frames per second

### 2.2 Leagues Included

| League | Country | Matches |
|--------|---------|---------|
| La Liga | Spain | ~125 |
| Premier League | England | ~104 |
| Serie A | Italy | ~105 |
| Bundesliga | Germany | ~61 |
| Ligue 1 | France | ~47 |
| Champions League | Europe | ~108 |

### 2.3 Corner Clip Extraction

**Process**:
1. Parse SoccerNet Labels-v2.json for corner kick events
2. Extract 30-second video clip around each corner
3. Validate clip integrity (remove corrupt files)

**Clip Structure**:
```
Total duration: 30 seconds
├── Observation window: 25 seconds (before corner)
└── Anticipation window: 5 seconds (after corner)
```

### 2.4 Statistics

| Metric | Value |
|--------|-------|
| Total clips | 4,836 |
| Valid clips | 4,690 (97%) |
| Clip duration | 30 seconds |
| Resolution | 720p (1280x720) |
| Frame rate | 25 fps |
| Frames per clip | 750 |
| Total size | 114 GB |

### 2.5 Frame Extraction

For FAANTRA training, frames are extracted and resized:

| Parameter | Value |
|-----------|-------|
| Output resolution | 224x224 |
| Format | JPEG |
| Frames extracted | 750 per clip |
| Subsampled to | 64 frames (for training) |

---

## 3. Label Distribution

### 3.1 Eight-Class Labels

| Class | Count | Percentage | Description |
|-------|-------|------------|-------------|
| NOT_DANGEROUS | 1,939 | 40.1% | Corner cleared, no threat created |
| CLEARED | 1,138 | 23.5% | Defensive clearance |
| SHOT_OFF_TARGET | 713 | 14.7% | Shot attempt missed goal |
| SHOT_ON_TARGET | 387 | 8.0% | Shot saved by goalkeeper |
| FOUL | 384 | 7.9% | Foul called during corner |
| GOAL | 172 | 3.6% | Goal scored from corner |
| OFFSIDE | 77 | 1.6% | Offside called |
| CORNER_WON | 26 | 0.5% | Another corner won |

### 3.2 Binary Labels

| Class | Includes | Count | Percentage |
|-------|----------|-------|------------|
| SHOT | GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET | 1,272 | 26.3% |
| NO_SHOT | All others | 3,556 | 73.7% |

### 3.3 Labeling Methodology

Labels assigned based on **immediate next event** after corner kick:
- If next event is a shot → SHOT (or specific shot type)
- If next event is clearance → CLEARED
- If next event is foul → FOUL
- etc.

This follows TacticAI (DeepMind) methodology for corner analysis.

---

## 4. Train/Val/Test Split

### 4.1 Split Sizes

| Split | Clips | Percentage |
|-------|-------|------------|
| Train | 3,868 | 80% |
| Valid | 483 | 10% |
| Test | 477 | 10% |

### 4.2 Split Strategy

- Random split at clip level
- No match-level stratification (clips from same match may appear in different splits)
- Class distribution preserved across splits

---

## 5. StatsBomb Freeze Frame Data

### 5.1 Source

**StatsBomb Open Data**: Free event-level soccer data
- Website: https://github.com/statsbomb/open-data
- Includes 360° freeze frame snapshots
- Player positions at moment of each event

### 5.2 Data Structure

Each corner kick event includes:

**Event Data**:
- Match ID, timestamp
- Corner taker (player, team)
- Corner location (x, y)

**360° Freeze Frame**:
- Position (x, y) of all visible players
- Team assignment (attacking/defending)
- Player role (if identifiable)

### 5.3 Coordinate System

StatsBomb pitch coordinates:
- X: 0 to 120 (defensive goal to attacking goal)
- Y: 0 to 80 (bottom sideline to top sideline)
- Corners at (120, 0) or (120, 80)

### 5.4 Statistics

| Metric | Value |
|--------|-------|
| Corners with freeze frames | 1,933 |
| Matches covered | 323 |
| Competitions | Multiple (Champions League, La Liga, etc.) |

### 5.5 Extracted Features

**Spatial features** (used for Classical ML):

| Feature | Description |
|---------|-------------|
| total_attacking | Number of attacking players |
| total_defending | Number of defending players |
| attacking_in_box | Attackers in penalty box |
| defending_in_box | Defenders in penalty box |
| attacking_density | Spatial concentration of attackers |
| defending_density | Spatial concentration of defenders |
| numerical_advantage | Attackers minus defenders |
| defending_depth | How deep defense is positioned |

---

## 6. Data Quality

### 6.1 Video Quality Issues

| Issue | Count | Resolution |
|-------|-------|------------|
| Corrupt source video | ~150 | Excluded from dataset |
| Missing frames | ~50 | Re-extracted |
| Wrong resolution | 0 | All 720p |

### 6.2 Label Quality Issues

| Issue | Resolution |
|-------|------------|
| Ambiguous outcomes | Used immediate next event |
| Missing labels | Excluded from dataset |
| Inconsistent timing | Normalized to corner start |

---

## 7. Data Access

### 7.1 File Locations

```
FAANTRA/data/
├── corners/
│   ├── corner_dataset.json    # Metadata (4,836 corners)
│   └── clips/                 # Video clips (114GB)
│       └── corner_XXXX/
│           └── 720p.mp4
├── corner_anticipation/       # Extracted frames (8-class)
│   ├── train/
│   ├── valid/
│   └── test/
└── corner_anticipation_binary/ # Binary labels
    ├── train/
    ├── valid/
    └── test/
```

### 7.2 Label Files

```
corner_anticipation/
├── class.txt              # Class names (8 classes)
├── train.json             # Train split metadata
├── val.json               # Valid split metadata
├── test.json              # Test split metadata
└── {split}/
    └── Labels-ball.json   # Per-clip labels
```

---

## 8. Preprocessing Pipeline

### 8.1 Video to Frames

```bash
# Extract frames from video clip
ffmpeg -i corner_XXXX/720p.mp4 \
       -vf "scale=224:224" \
       -q:v 2 \
       corner_XXXX/frame_%04d.jpg
```

### 8.2 Label Conversion (8-class to Binary)

```python
# Mapping
SHOT_CLASSES = {'GOAL', 'SHOT_ON_TARGET', 'SHOT_OFF_TARGET'}
NO_SHOT_CLASSES = {'CLEARED', 'NOT_DANGEROUS', 'FOUL', 'OFFSIDE', 'CORNER_WON'}

# Convert
if original_label in SHOT_CLASSES:
    binary_label = 'SHOT'
else:
    binary_label = 'NO_SHOT'
```

---

## 9. Limitations

### 9.1 Video Data Limitations

- **Broadcast view only**: No tactical camera angles
- **Variable quality**: Some matches have compression artifacts
- **Single viewpoint**: Cannot see all players simultaneously

### 9.2 Freeze Frame Limitations

- **Single snapshot**: Only captures one moment in time
- **Incomplete coverage**: Not all corners have freeze frames
- **Position only**: No velocity or movement direction

### 9.3 Label Limitations

- **Subjective categories**: NOT_DANGEROUS vs CLEARED overlap
- **Rare classes**: CORNER_WON only 0.5% of data
- **No causal labels**: Cannot know WHY outcome occurred
