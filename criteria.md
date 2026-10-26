# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1. Overall Accuracy ≥ 0.65

Test set accuracy must be at least 0.65 (65%) across all three random seeds.

**Why this target:**

With three balanced labels, random guessing achieves ~33% accuracy. An accuracy of 65% means the model makes meaningful distinctions roughly twice as often as random chance. This is the minimum threshold where the classifier becomes useful for real application. Below this, I'd question whether it learned genuine differences between ANALYSIS, HOT_TAKE, and DISCUSSION. Given these are somewhat subjective text categories (not hard facts), 65% is ambitious but achievable with 200 examples.



---

## 2. Per-Label F1 ≥ 0.55 for all labels

Every label (ANALYSIS, HOT_TAKE, DISCUSSION) must have F1 ≥ 0.55 on the test set, across all three seeds.

**Why this target:**

I want the model to handle all three labels competently, not excel at one while failing at others. F1 balances precision and recall, so 0.55 means the model neither over-predicts nor under-predicts any label catastrophically. I'm setting this slightly lower than overall accuracy (0.55 vs 0.65) because some labels will be easier than others - DISCUSSION likely has clearer signals (question marks, invitation language) while ANALYSIS vs HOT_TAKE sits on a subtler boundary. Requiring 0.55 minimum ensures no label is a total failure.



---

## 3. Label Balance: No label >60% of dataset

In labels.csv, each label (ANALYSIS, HOT_TAKE, DISCUSSION) must be between 20-60% of the total.

**Why this target:**

If one label dominates (>60%), the model can achieve decent accuracy by just guessing that label most of the time - it wouldn't be learning real distinctions. I'm aiming for roughly balanced labels (ideally 33/33/33) because that forces the model to learn all three categories equally. The 20-60% range allows for natural imbalance in r/movies posts while preventing any label from being either rare (< 20%, too few examples to learn) or overwhelming (>60%, becomes the default guess). This is stricter than the notebook's 70% warning because I want forced diversity in my training data.



---

## 4. Inter-Annotator Agreement ≥ 0.75 on cold-labeled sample

On the 20 posts I labeled "cold" (no AI assistance), a second annotator using only my written definitions should agree with me at least 75% of the time (15+ out of 20 posts).

**Why this target:**

If I can't get consistent labeling on my own definitions, the model has no hope of learning them. This criterion tests whether my label boundaries are clear enough for someone else to apply. Agreement of 75% means disagreement on only 5 out of 20 posts, which feels like reasonable inter-human consistency for somewhat subjective categories like "analysis" vs "hot take." Below 75%, my definitions are too vague or my own application is inconsistent. This is measured on the cold-labeled subset specifically because those show my unassisted judgment.



---

## 5. Confidence Calibration: High-confidence predictions ≥ 0.75 accurate

When the model's confidence (max softmax probability) is ≥ 0.80, it should be correct at least 75% of the time on the test set.

**Why this target:**

A model that's confidently wrong is worse than one that admits uncertainty. When the model says "I'm 80%+ sure this is ANALYSIS," I want to trust that prediction - it should be right significantly more often than the baseline 65% accuracy. The 75% threshold means confident predictions are about 10 percentage points more reliable than average, which makes the confidence score actionable (I can prioritize reviewing low-confidence predictions). I'm not demanding perfection (100%) because even human labelers disagree on borderline posts, but confident predictions should be meaningfully more accurate.



---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
