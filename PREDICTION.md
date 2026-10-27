# Baseline vs Trained Model Prediction

**Prediction (written before seeing baseline results):**

The trained model will beat the baseline on overall accuracy and all three per-label F1 scores. 

**Reasoning:**

The trained model was fine-tuned on 140 examples that demonstrated the specific patterns of each label:
- ANALYSIS posts consistently included checkable facts (box office numbers, production details, named comparisons)
- HOT_TAKE posts used strong declarative language without supporting evidence  
- DISCUSSION posts explicitly invited input through questions or requests

The baseline must infer these distinctions from text definitions alone, without seeing examples of how the patterns actually manifest in r/movies posts. The trained model learned from examples, so it should recognize the subtle signals (like "What do you think?" for DISCUSSION or "$82M worldwide" for ANALYSIS) that definitions can't fully capture.

**Specific expectation:** Trained model will have 15-25 percentage points higher accuracy than baseline, with the biggest gap on ANALYSIS vs HOT_TAKE distinction (the hardest boundary).
