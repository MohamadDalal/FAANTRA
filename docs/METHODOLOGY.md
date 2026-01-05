# Methodology: Corner Kick Outcome Prediction

This document describes all methods used for predicting corner kick outcomes from broadcast video.

---

## 1. Problem Definition

**Task**: Predict what happens after a corner kick is taken, using only video observed BEFORE the kick.

**Why This Is Hard**: The outcome (shot, clearance, goal) depends on:
- Player movements AFTER the corner
- Aerial duels and headers
- Goalkeeper decisions
- Defensive positioning changes

These events occur in the "anticipation window" (after observation ends), making prediction fundamentally difficult.

---

## 2. FAANTRA Model Architecture

**FAANTRA** = Football Action ANticipation TRAnsformer

**Source**: Zhou et al., "Adversarial Learning for Feature Shift Detection and Correction" (SoccerNet Ball Action Anticipation Challenge)

### 2.1 Overall Architecture

```
Input Video Frames (224x224 RGB)
         ↓
┌─────────────────────────────┐
│  Feature Backbone           │
│  (RegNetY-004 + GSF/GSM)    │
│  Output: 512-dim per frame  │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Transformer Encoder        │
│  - 2 layers                 │
│  - 8 attention heads        │
│  - 512 hidden dimension     │
│  - Masked self-attention    │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Transformer Decoder        │
│  - 2 layers                 │
│  - 4 learned queries        │
│  - Cross-attention to encoder│
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Prediction Heads           │
│  - Class prediction (softmax)│
│  - Temporal offset (regression)│
│  - Actionness score         │
└─────────────────────────────┘
```

### 2.2 Feature Backbone: RegNetY-004 with GSF/GSM

**RegNetY-004**: Efficient CNN backbone pretrained on ImageNet
- 4.0 GFLOPs computational cost
- Outputs 512-dimensional feature vector per frame

**GSF (Gated Shift Fuse)**: Temporal modeling module
- Shifts feature channels across adjacent frames
- Captures short-term temporal patterns
- Enables frame-to-frame motion understanding

**GSM (Gated Shift Module)**: Variant of GSF
- Applied at multiple network stages
- Fold dimensions: 28/104, 52/208, 112/440 (from logs)

### 2.3 Transformer Architecture

**Encoder**:
- 2 transformer layers
- 8 attention heads
- 512 hidden dimension
- Masked self-attention (window size 15)
- Processes observed frames sequentially

**Decoder**:
- 2 transformer layers
- 4 learned query tokens
- Each query predicts one potential action
- Cross-attention to encoder output
- Masked self-attention (window size 19)

### 2.4 Prediction Heads

**Classification Head**:
- Softmax over N classes + background
- Predicts action type (SHOT, NO_SHOT, etc.)

**Temporal Offset Head**:
- Regression head
- Predicts when action occurs (in seconds)

**Actionness Head**:
- Binary classifier
- Predicts if query corresponds to real action vs background

---

## 3. Training Configuration

### 3.1 Hyperparameters (8-Class Model)

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 2 | Samples per gradient update |
| learning_rate | 0.0001 | Adam optimizer LR |
| weight_decay | 0.005 | L2 regularization |
| num_epochs | 50 | Training iterations |
| warm_up_epochs | 5 | LR warmup period |
| clip_len | 64 | Frames per clip (subsampled) |
| obs_perc | 0.5 | 50% observation, 50% anticipation |
| hidden_dim | 512 | Transformer dimension |

### 3.2 Hyperparameters (Binary Model)

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 1 | Reduced due to GPU memory |
| num_epochs | 30 | Fewer epochs needed |
| class_weights | [0.1, 2.8, 1.0] | [BG, SHOT, NO_SHOT] |

### 3.3 Class Weights

Weights compensate for class imbalance:

**8-Class**: `[0.1, 11.0, 5.0, 2.7, 80.0, 1.7, 1.0, 5.0, 25.0]`
- Background: 0.1 (very common)
- CORNER_WON: 80.0 (very rare, 0.5%)
- NOT_DANGEROUS: 1.0 (baseline, 40%)

**Binary**: `[0.1, 2.8, 1.0]`
- Background: 0.1
- SHOT: 2.8 (26% of data)
- NO_SHOT: 1.0 (74% of data)

### 3.4 Loss Function

**Multi-task loss**:
```
L_total = L_cls + λ_offset * L_offset + L_actionness
```

Where:
- `L_cls`: Cross-entropy loss for class prediction
- `L_offset`: L1 loss for temporal offset (λ=10)
- `L_actionness`: Binary cross-entropy for actionness

---

## 4. Evaluation Metric: mAP@∞

### 4.1 Definition

**mAP@∞** = mean Average Precision with infinite tolerance window

This metric measures how well the model predicts actions that occur in the anticipation window, regardless of exact timing.

### 4.2 Calculation

1. **For each class**:
   - Rank all predictions by confidence score
   - For each prediction, check if a ground truth action exists in anticipation window
   - Calculate precision at each recall threshold
   - Average precision = area under precision-recall curve

2. **Mean over classes**:
   ```
   mAP@∞ = (1/N) * Σ AP_class
   ```

### 4.3 Tolerance Windows

| Metric | Tolerance | Meaning |
|--------|-----------|---------|
| mAP@1 | 1 second | Prediction within 1s of ground truth |
| mAP@2 | 2 seconds | Prediction within 2s of ground truth |
| mAP@∞ | Infinite | Any prediction in anticipation window |

### 4.4 Why mAP@∞ for Corners

Corner outcomes occur at variable times (1-5 seconds after kick). Using mAP@∞ measures whether the model predicts the CORRECT outcome, regardless of exact timing.

---

## 5. Classical ML Baseline (Research Project)

### 5.1 Data Source

**StatsBomb Open Data**:
- 360° freeze frame data at moment of corner kick
- Player positions (x, y) for all 22 players
- Available for ~1,900 corners with freeze frames

### 5.2 Features Used (After Removing Leakage)

**Valid Features** (known at corner time):

| Feature | Description |
|---------|-------------|
| corner_x, corner_y | Corner kick starting position |
| minute, second | Match time |
| total_attacking | Attacking players in frame |
| total_defending | Defending players in frame |
| attacking_in_box | Attackers in penalty box |
| defending_in_box | Defenders in penalty box |
| attacking_density | Spatial concentration of attackers |
| defending_density | Spatial concentration of defenders |
| numerical_advantage | Attackers minus defenders |

**Leaked Features** (removed):
- pass_end_x, pass_end_y (actual ball landing = outcome)
- is_shot_assist (literally the target variable)
- duration (only known after event)

### 5.3 Models Tested

| Model | Implementation |
|-------|----------------|
| Random Forest | sklearn, 100 trees |
| XGBoost | xgboost, default params |
| MLP | sklearn, 2 hidden layers |

### 5.4 Results

**With Leakage** (invalid):
- Random Forest: 87.97% accuracy, 0.85 AUC

**Without Leakage** (valid):
- Best model: 71.32% accuracy, **0.43 AUC**
- Essentially random (AUC 0.5 = random guessing)

---

## 6. Comparison: Video vs Spatial Data

### 6.1 Why Compare mAP vs AUC?

Both metrics measure ranking quality:
- **AUC**: Area under ROC curve (binary classification)
- **mAP**: Area under precision-recall curve (detection)

For binary classification:
- AUC = 0.5 means random
- mAP = 0.5 means random (for 50/50 class balance)

Our binary split is 26%/74%, so:
- Random classifier mAP ≈ 0.5 (weighted average)

### 6.2 Results Comparison

| Approach | Data | Metric | Result | Interpretation |
|----------|------|--------|--------|----------------|
| Classical ML | StatsBomb freeze frames | AUC | 0.43 | Random |
| FAANTRA | Broadcast video | mAP@∞ | 0.50 | Random |

### 6.3 Conclusion

**Both approaches achieve random performance** for binary shot prediction.

This confirms:
1. Pre-corner information (positions, video) cannot predict shots
2. The outcome depends on post-corner events
3. Shot prediction is fundamentally unpredictable from observation data

---

## 7. Data Pipeline

### 7.1 Video Processing

```
SoccerNet Videos (720p, 25fps)
         ↓
Extract 30-second clips around corner events
         ↓
4,836 corner clips (114GB total)
         ↓
Extract frames (750 frames per clip)
         ↓
Resize to 224x224 for FAANTRA
```

### 7.2 Label Processing

**8-Class Labels**:
```
GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET,
CORNER_WON, CLEARED, NOT_DANGEROUS, FOUL, OFFSIDE
```

**Binary Labels** (mapped from 8-class):
```
SHOT = {GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET}
NO_SHOT = {CORNER_WON, CLEARED, NOT_DANGEROUS, FOUL, OFFSIDE}
```

### 7.3 Train/Val/Test Split

| Split | Clips | Percentage |
|-------|-------|------------|
| Train | 3,868 | 80% |
| Valid | 483 | 10% |
| Test | 477 | 10% |

---

## 8. Key References

1. **FAANTRA**: SoccerNet Ball Action Anticipation Challenge baseline
   - Task: Anticipate ball actions (pass, shot, drive) from video

2. **TacticAI** (DeepMind, 2024): Corner kick analysis using GNNs
   - Used "immediate next event" labeling approach

3. **StatsBomb Open Data**: Free event and 360° data
   - Source for classical ML experiments

4. **SoccerNet**: Large-scale soccer video dataset
   - Source for broadcast video clips

---

## 9. Reproducibility

### 9.1 Code Locations

| Component | Path |
|-----------|------|
| FAANTRA model | `FAANTRA/model/futr.py` |
| Training script | `FAANTRA/main.py` |
| Evaluation | `FAANTRA/test.py`, `FAANTRA/eval_BAA.py` |
| 8-class config | `FAANTRA/config/SoccerNetBall/Corner-Config.json` |
| Binary config | `FAANTRA/config/SoccerNetBall/Corner-Binary.json` |
| Label conversion | `FAANTRA/scripts/convert_to_binary_labels.py` |

### 9.2 Trained Checkpoints

| Model | Path |
|-------|------|
| 8-class | `checkpoints/corners/corner_baselinemodel/transformer/checkpoint/checkpoint.ckpt` |
| Binary | `checkpoints/corners-binary/corner_binary/transformer/checkpoint/checkpoint.ckpt` |

### 9.3 Hardware

- GPU: NVIDIA V100 32GB / A100 80GB
- Training time: ~24 hours (8-class), ~12 hours (binary)
- SLURM cluster with module system
