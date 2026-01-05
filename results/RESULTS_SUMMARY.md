# FAANTRA Corner Kick Prediction Results

## Overview

This document summarizes all FAANTRA model experiments on corner kick prediction from SoccerNet broadcast videos.

## Dataset

- **Source**: SoccerNet broadcast videos (550 games, 6 leagues)
- **Total corners**: 4,836 clips
- **Train/Val/Test split**: 3,868 / 483 / 477 clips
- **Clip duration**: 30 seconds (25s observation + 5s anticipation)
- **Frame extraction**: 224x224 @ 25fps

## Experiments

### 1. Eight-Class Outcome Prediction

**Task**: Predict corner kick outcome from 8 possible classes.

| Class | Count | % |
|-------|-------|---|
| NOT_DANGEROUS | 1,939 | 40.1% |
| CLEARED | 1,138 | 23.5% |
| SHOT_OFF_TARGET | 713 | 14.7% |
| SHOT_ON_TARGET | 387 | 8.0% |
| FOUL | 384 | 7.9% |
| GOAL | 172 | 3.6% |
| OFFSIDE | 77 | 1.6% |
| CORNER_WON | 26 | 0.5% |

**Results**:
- **mAP@infinity**: 12.58%
- **Frame-level error**: 9.38%

| Class | mAP@infinity |
|-------|--------------|
| OFFSIDE | 38.36% |
| NOT_DANGEROUS | 25.76% |
| FOUL | 12.79% |
| CORNER_WON | 10.48% |
| SHOT_OFF_TARGET | 7.97% |
| CLEARED | 2.52% |
| SHOT_ON_TARGET | 2.10% |
| GOAL | 0.63% |

**Checkpoint**: `checkpoints/corners/corner_baselinemodel/transformer/checkpoint/checkpoint.ckpt`

---

### 2. Binary Shot/No-Shot Prediction

**Task**: Predict whether corner results in a shot attempt or not.

| Class | Mapping | Count | % |
|-------|---------|-------|---|
| SHOT | GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET | 1,272 | 26.3% |
| NO_SHOT | CLEARED, NOT_DANGEROUS, FOUL, OFFSIDE, CORNER_WON | 3,556 | 73.7% |

**Results**:
- **mAP@infinity**: 50.0% (essentially random)
- **Validation accuracy**: 85.7%
- **Frame-level error**: 9.38%
- **Training epochs**: 30

| Class | mAP@infinity |
|-------|--------------|
| NO_SHOT | 74.21% |
| SHOT | 25.79% |

**Checkpoint**: `checkpoints/corners-binary/corner_binary/transformer/checkpoint/checkpoint.ckpt`

---

## Comparison with Research Project

| Approach | Data Source | Task | Result |
|----------|-------------|------|--------|
| Classical ML | StatsBomb freeze frames | Binary shot | AUC = 0.43 |
| FAANTRA | Broadcast video | Binary shot | **mAP = 50%** |
| FAANTRA | Broadcast video | 8-class outcome | mAP = 12.6% |

## Baseline Comparison

| Model | Task | mAP@infinity |
|-------|------|--------------|
| FAANTRA (paper) | Ball Action Anticipation | 26-28% |
| FAANTRA (ours) | Ball Action Anticipation | 18.48% |
| FAANTRA (ours) | Corner 8-class | 12.58% |
| FAANTRA (ours) | Corner binary | 50.0% |

## Key Findings

1. **Shot prediction is fundamentally unpredictable**: Both video-based (FAANTRA) and spatial-data-based (Classical ML) approaches achieve essentially random performance (~50% mAP / 0.43 AUC) for binary shot prediction.

2. **Outcome depends on post-corner events**: The corner kick outcome is determined by events that happen AFTER the corner is taken (player movements, aerial duels, goalkeeper decisions), which are not observable in the pre-corner video.

3. **8-class prediction shows some signal**: The model learns meaningful patterns for some classes (OFFSIDE: 38%, NOT_DANGEROUS: 26%), suggesting certain outcomes may have visual precursors.

4. **Task complexity matters**: Ball action anticipation (18.48% mAP) is easier than corner outcome prediction (12.6% mAP) because ball actions have visible cues in the observation window.

## Files

- `results/corners.json` - 8-class results
- `results/corners-binary.json` - Binary results
- `config/SoccerNetBall/Corner-Config.json` - 8-class config
- `config/SoccerNetBall/Corner-Binary.json` - Binary config

## Date

Results generated: 2026-01-04
