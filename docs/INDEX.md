# Documentation Index: Corner Kick Prediction Project

This index provides a structured overview of all project documentation.

---

## Quick Reference

| Question | Document |
|----------|----------|
| What are the results? | [RESULTS.md](RESULTS.md) |
| How does the model work? | [METHODOLOGY.md](METHODOLOGY.md) |
| How do I run the code? | [../README.md](../README.md) or root CLAUDE.md |
| What data was used? | [DATASET.md](DATASET.md) |

---

## Document Descriptions

### 1. METHODOLOGY.md
**Purpose**: Explains all methods used in the project

**Contents**:
- FAANTRA model architecture (transformer, RegNetY backbone)
- Training configuration (hyperparameters, loss functions)
- Evaluation metric (mAP@∞) explanation
- Classical ML baseline methodology
- Comparison between video and spatial approaches

**Use when**: Writing methods section of report/thesis

---

### 2. RESULTS.md
**Purpose**: Presents all experimental results

**Contents**:
- 8-class outcome prediction results (12.6% mAP)
- Binary shot/no-shot results (50% mAP)
- Classical ML comparison (0.43 AUC)
- Per-class performance breakdown
- Error analysis and interpretation

**Use when**: Writing results section of report/thesis

---

### 3. DATASET.md
**Purpose**: Describes data sources and processing

**Contents**:
- SoccerNet video source (550 games, 6 leagues)
- Corner clip extraction (4,836 clips, 114GB)
- Label distribution and class definitions
- Train/val/test split details
- StatsBomb freeze frame data

**Use when**: Writing data section of report/thesis

---

## Key Numbers for Reports

### Dataset
- **Total corners**: 4,836 video clips
- **Video duration**: 30 seconds per clip (25s obs + 5s antic)
- **Frame resolution**: 224x224 @ 25fps
- **Total size**: 114GB

### Results
| Experiment | Result | Meaning |
|------------|--------|---------|
| 8-class FAANTRA | 12.6% mAP | Better than random |
| Binary FAANTRA | 50% mAP | Random chance |
| Binary Classical ML | 0.43 AUC | Random chance |

### Model
- **Architecture**: Transformer (2 enc + 2 dec layers)
- **Backbone**: RegNetY-004 + GSF
- **Hidden dim**: 512
- **Training**: 50 epochs (8-class), 30 epochs (binary)

---

## File Locations

### Documentation
```
FAANTRA/
├── docs/
│   ├── INDEX.md          # This file
│   ├── METHODOLOGY.md    # Methods explanation
│   ├── RESULTS.md        # Experimental results
│   └── DATASET.md        # Data description
└── results/
    ├── corners.json      # 8-class metrics (JSON)
    └── corners-binary.json # Binary metrics (JSON)
```

### Code
```
FAANTRA/
├── main.py               # Training entry point
├── test.py               # Evaluation entry point
├── model/futr.py         # FAANTRA model definition
├── dataset/frame.py      # Data loading
└── config/SoccerNetBall/
    ├── Corner-Config.json    # 8-class config
    └── Corner-Binary.json    # Binary config
```

### Checkpoints
```
FAANTRA/checkpoints/
├── corners/              # 8-class model
└── corners-binary/       # Binary model
```

---

## Citation Information

If using this work, cite:

**FAANTRA** (base model):
- SoccerNet Ball Action Anticipation Challenge
- https://www.soccer-net.org/

**SoccerNet** (video data):
- Deliège et al., "SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos"

**StatsBomb** (freeze frame data):
- StatsBomb Open Data: https://github.com/statsbomb/open-data

---

## Report Writing Checklist

For a complete report, ensure you cover:

- [ ] **Introduction**: Corner kick prediction problem
- [ ] **Related Work**: FAANTRA, TacticAI, SoccerNet
- [ ] **Data**: SoccerNet videos + StatsBomb freeze frames
- [ ] **Methods**: FAANTRA architecture, Classical ML baseline
- [ ] **Experiments**: 8-class, binary, cross-method comparison
- [ ] **Results**: mAP@∞, AUC, per-class breakdown
- [ ] **Discussion**: Why shot prediction is random
- [ ] **Conclusion**: Both approaches fail equally
