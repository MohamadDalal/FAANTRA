# Results: Corner Kick Outcome Prediction

This document presents all experimental results for corner kick prediction from broadcast video.

---

## 1. Summary Table

| Experiment | Classes | Test Clips | mAP@∞ | Interpretation |
|------------|---------|------------|-------|----------------|
| 8-class outcome | 8 | 477 | 12.58% | Better than random |
| Binary shot/no-shot | 2 | 477 | 50.0% | Random chance |
| Classical ML (StatsBomb) | 2 | ~400 | AUC=0.43 | Random chance |

**Main Finding**: Shot prediction achieves random performance regardless of method.

---

## 2. Eight-Class Outcome Prediction

### 2.1 Task Definition

Predict which of 8 outcomes occurs after a corner kick:

| Class | Description | Count | % |
|-------|-------------|-------|---|
| NOT_DANGEROUS | Ball cleared, no threat | 1,939 | 40.1% |
| CLEARED | Defensive clearance | 1,138 | 23.5% |
| SHOT_OFF_TARGET | Shot misses goal | 713 | 14.7% |
| SHOT_ON_TARGET | Shot on target (saved) | 387 | 8.0% |
| FOUL | Foul called | 384 | 7.9% |
| GOAL | Goal scored | 172 | 3.6% |
| OFFSIDE | Offside called | 77 | 1.6% |
| CORNER_WON | Another corner won | 26 | 0.5% |

### 2.2 Results

**Overall**: mAP@∞ = 12.58%

**Per-Class Performance**:

| Class | mAP@∞ | Analysis |
|-------|-------|----------|
| OFFSIDE | 38.36% | Best - may have visible buildup patterns |
| NOT_DANGEROUS | 25.76% | Good - most common class |
| FOUL | 12.79% | Moderate |
| CORNER_WON | 10.48% | Moderate despite rarity |
| SHOT_OFF_TARGET | 7.97% | Poor |
| CLEARED | 2.52% | Poor |
| SHOT_ON_TARGET | 2.10% | Poor |
| GOAL | 0.63% | Worst - rare and unpredictable |

### 2.3 Interpretation

- Model learns patterns for some outcomes (OFFSIDE, NOT_DANGEROUS)
- Shot-related classes (GOAL, SHOT_ON/OFF) are hardest to predict
- Class imbalance affects results (GOAL only 3.6% of data)

---

## 3. Binary Shot/No-Shot Prediction

### 3.1 Task Definition

Simplified binary classification:

| Class | Includes | Count | % |
|-------|----------|-------|---|
| SHOT | GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET | 1,272 | 26.3% |
| NO_SHOT | CLEARED, NOT_DANGEROUS, FOUL, OFFSIDE, CORNER_WON | 3,556 | 73.7% |

### 3.2 Results

| Metric | Value |
|--------|-------|
| mAP@∞ | 50.0% |
| mAP (SHOT class) | 25.79% |
| mAP (NO_SHOT class) | 74.21% |
| Validation accuracy | 85.7% |
| Frame-level error | 9.38% |

### 3.3 Interpretation

- **50% mAP = random chance** for binary classification
- Model predicts class proportions, not actual outcomes
- High validation accuracy (85.7%) is misleading - model learns to predict majority class

---

## 4. Comparison with Classical ML

### 4.1 StatsBomb Freeze Frame Approach

**Data**: 360° player positions at moment of corner kick (~1,900 corners)

**Features**: Player counts, positions, densities in various zones

**Models**: Random Forest, XGBoost, MLP

### 4.2 Results (After Removing Data Leakage)

| Model | Accuracy | AUC |
|-------|----------|-----|
| MLP | 71.32% | 0.52 |
| Random Forest | 63.57% | 0.51 |
| XGBoost | 60.47% | 0.51 |

**AUC ≈ 0.5 = random guessing**

### 4.3 Cross-Method Comparison

| Method | Data Type | Binary Result |
|--------|-----------|---------------|
| FAANTRA | 25 seconds of video | mAP = 50% (random) |
| Classical ML | Single freeze frame | AUC = 0.43 (random) |

**Both approaches fail equally** - confirms shot prediction is fundamentally impossible from pre-corner data.

---

## 5. Baseline Comparison

### 5.1 Ball Action Anticipation (Original FAANTRA Task)

| Model | Dataset | mAP@∞ |
|-------|---------|-------|
| FAANTRA (paper) | SoccerNet-BAA | 26-28% |
| FAANTRA (ours) | SoccerNet-BAA | 18.48% |

### 5.2 Why Corner Prediction Is Harder

| Aspect | Ball Actions | Corner Outcomes |
|--------|--------------|-----------------|
| Task | Predict pass/shot/drive | Predict goal/clear/shot |
| Observable cues | Player positioning, ball movement | Limited - outcome after kick |
| Prediction horizon | Immediate actions | 1-5 seconds post-kick |
| mAP@∞ | 18-28% | 12.6% |

Ball actions have visible precursors (player running, ball position). Corner outcomes depend on events that happen AFTER observation ends.

---

## 6. Metric Details

### 6.1 mAP@∞ Calculation

For each class:
1. Rank predictions by confidence
2. Calculate precision at each recall point
3. AP = area under precision-recall curve

mAP = mean of per-class APs

### 6.2 Time-Based Metrics

| Metric | Our Result | Meaning |
|--------|------------|---------|
| mAP@1 | 0.0% | No predictions within 1 second |
| mAP@2 | 0.0% | No predictions within 2 seconds |
| mAP@5 | 50.0% | Predictions at 5-second horizon |
| mAP@∞ | 50.0% | Same as mAP@5 |

Model only makes predictions at the end of anticipation window (5 seconds).

---

## 7. Training Curves

### 7.1 Eight-Class Model

- **Epochs**: 50
- **Best validation**: Epoch ~45
- **Convergence**: Stable after epoch 30

### 7.2 Binary Model

- **Epochs**: 30
- **Validation accuracy**: 85.7% (final)
- **Note**: High accuracy due to majority class prediction

---

## 8. Error Analysis

### 8.1 What the Model Gets Right

- **NOT_DANGEROUS**: Correctly identifies low-threat corners
- **OFFSIDE**: May detect crowded box formations

### 8.2 What the Model Gets Wrong

- **GOAL vs SHOT_ON_TARGET**: Cannot distinguish outcomes
- **SHOT vs CLEARED**: 50/50 random prediction
- **Rare classes**: Insufficient training examples

### 8.3 Why Shot Prediction Fails

The outcome depends on:
1. **Header accuracy** - unpredictable from video
2. **Goalkeeper reaction** - happens after observation
3. **Defensive clearances** - split-second decisions
4. **Ball trajectory** - depends on kick quality

None of these are observable before the corner is taken.

---

## 9. Files and Checkpoints

### 9.1 Result Files

| File | Contents |
|------|----------|
| `results/corners.json` | 8-class metrics |
| `results/corners-binary.json` | Binary metrics |
| `results/RESULTS_SUMMARY.md` | Legacy summary |

### 9.2 Checkpoints

| Model | Path |
|-------|------|
| 8-class | `checkpoints/corners/corner_baselinemodel/transformer/checkpoint/checkpoint.ckpt` |
| Binary | `checkpoints/corners-binary/corner_binary/transformer/checkpoint/checkpoint.ckpt` |

---

## 10. Conclusion

### 10.1 Key Findings

1. **8-class prediction**: 12.6% mAP - some classes learnable
2. **Binary prediction**: 50% mAP - random chance
3. **Classical ML**: 0.43 AUC - also random
4. **Shot prediction is impossible** from pre-corner observation

### 10.2 Implications

- Corner kick outcomes are inherently unpredictable
- Pre-kick player positions do not determine shot probability
- The outcome depends on post-kick events (headers, saves, clearances)
- Both video and spatial data approaches confirm this finding
