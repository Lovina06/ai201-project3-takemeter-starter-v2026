# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

<!-- Your community, and what your classifier sorts posts into. Three or four
     sentences. -->

TakeMeter classifies movie discussion posts from r/movies into three categories based on how they engage with film criticism. ANALYSIS posts back claims with specific, checkable facts (box office numbers, production details, named comparisons). HOT_TAKE posts make confident judgments with no supporting evidence. DISCUSSION posts explicitly invite conversation through questions or requests for input. The classifier helps identify which posts contribute substantive analysis versus unsupported opinion versus conversation-seeking.



---

## Label Taxonomy

<!-- Each label: a one-sentence definition and two real examples from your
     reading. Then your decision rule for the hardest boundary.

     The decision rule is worth a point on its own and it's the thing most
     people leave out. Every taxonomy has a hardest boundary. Name yours. -->

### `ANALYSIS`

**Definition:** Makes an argument backed by at least one specific, checkable fact (box office numbers, release dates, production details, named crew/cast, critical scores, specific plot points, technical details like runtime or budget).

**Example 1:**
> "Dune Part Two made $82M in its opening weekend, which is impressive considering the first film only made $41M. The studio clearly built better word-of-mouth this time, and the expanded IMAX release helped - IMAX alone accounted for $28M of that total."

**Example 2:**
> "The Substance uses body horror in the Cronenberg tradition - both films feature slow physical transformation as metaphor for social pressure. The makeup effects here rival Videodrome's practical work, which is saying something given today's CGI shortcuts."

### `HOT_TAKE`

**Definition:** A confident claim or judgment with NO specific supporting evidence. May be true or insightful, but isn't argued - just stated. Typically uses strong language ("best," "worst," "masterpiece," "garbage") without backing it up.

**Example 1:**
> "Wicked is genuinely one of the best musicals ever put to film. The performances, the visuals, everything works. If you disagree you probably just don't like musicals."

**Example 2:**
> "Anyone saying The Wild Robot is mid clearly has no taste. That movie hits different and if you know you know."

### `DISCUSSION`

**Definition:** Explicitly invites others to weigh in. May ask a question, request recommendations, seek interpretation help, or present multiple viewpoints and ask which is right. The PURPOSE is conversation, not declaration.

**Example 1:**
> "I just finished Anora and I'm genuinely unsure if the ending is hopeful or devastating. What did you all take from it? Was she better off or worse?"

**Example 2:**
> "Looking for movies with unreliable narrators like Shutter Island or Fight Club. Preferably where the twist actually changes how you see the whole story, not just tacked on. What hits that mark for you?"

### The hardest boundary

**Which two labels:** ANALYSIS vs HOT_TAKE when strong opinions meet evidence

**The decision rule I used every time:** If the post names at least one specific, checkable fact (a comparison to another named work, a technical spec, a named person/number/date), it's ANALYSIS even if the tone is confident. Hot takes are ONLY for posts with zero specific grounding - pure opinion statements.



---

## The Dataset

<!-- Where you collected from, how you labelled, your counts, and three hard
     cases. -->

**Where the posts came from:** Generated realistic r/movies-style discussions based on actual 2024-2026 film releases and typical community patterns. Posts reference real box office numbers, production details, and genuine discourse topics from movie communities.

**How I labelled them:** Labeled 20 posts cold first (no AI assistance) to establish my own consistency. Then generated remaining 180 posts using AI with my label definitions, manually reviewing and correcting each one to ensure they matched my decision rules. All pre-labeled posts are marked as such in the notes column.

**Counts per label:**
- ANALYSIS: 68 (34.0%)
- HOT_TAKE: 67 (33.5%)
- DISCUSSION: 65 (32.5%)

| Label | Count | Share |
|---|---|---|
| ANALYSIS | 68 | 34.0% |
| HOT_TAKE | 67 | 33.5% |
| DISCUSSION | 65 | 32.5% |
| **Total** | 200 | 100% |

**Three hard cases**

<!-- Any post that made you pause: what it was, which two labels it could have
     been, and what you chose. These are worth more than the easy 190. -->

**1.**
> *The post:* "Oppenheimer is Nolan's best work. The sound design alone puts it above Dunkirk - every explosion feels visceral in a way Dunkirk's beach scenes didn't. The 70mm IMAX presentation is essential."
>
> *Could have been:* ANALYSIS or HOT_TAKE
>
> *I chose ANALYSIS, because:* Although it opens with the declarative "best work" (HOT_TAKE signal), it immediately supports the claim with specific comparisons: Dunkirk by name, sound design details, and 70mm IMAX format mention. My decision rule says if a post names at least one checkable fact, it's ANALYSIS even with confident tone.

**2.**
> *The post:* "Does anyone else think A24 horror is getting too pretentious? Like I appreciate slow burns but at what point does artsy become boring? Where's the line for you?"
>
> *Could have been:* DISCUSSION or HOT_TAKE
>
> *I chose DISCUSSION, because:* While it contains strong opinions ("pretentious," "boring"), it's framed entirely as questions seeking others' perspectives. The "Does anyone else think..." opening and "Where's the line for you?" closing explicitly invite conversation, making the PURPOSE discussion not declaration.

**3.**
> *The post:* "Nope made $174M worldwide on a $68M budget which is solid but lower than Get Out's $255M and Us's $256M. Peele said it's his most personal film and the spectacle vs narrative balance was intentional."
>
> *Could have been:* ANALYSIS (certain, but interesting nuance)
>
> *I chose ANALYSIS, because:* This was actually straightforward - multiple specific numbers, named films for comparison, direct Peele quote. The interesting part was recognizing that ANALYSIS doesn't require making an argument, just presenting facts. This post informs rather than persuades, but that still counts as analysis when backed by specifics.

---

## The Training Run

<!-- Your starting model, your settings, and anything you changed and why. -->

**Base model:** distilbert-base-uncased (DistilBERT, 66M parameters, distilled from BERT-base)

**Settings:** 
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Seed: 42
- Max sequence length: 128 tokens

**Anything I changed from the defaults, and why:**

No changes from notebook defaults. The default settings (3 epochs, 2e-5 learning rate) are standard for fine-tuning DistilBERT on small text classification tasks. With only 140 training examples, 3 epochs provides enough iterations without overfitting risk. The 128 token limit is sufficient since movie discussion posts average ~50-80 tokens.

**Split sizes:**

- Train: 140 examples (70%)
- Validation: 30 examples (15%)
- Test: 30 examples (15%)

Per-label counts in test split:
- ANALYSIS: 10 examples
- HOT_TAKE: 10 examples  
- DISCUSSION: 10 examples

Perfect balance in test split (10/10/10) due to stratified sampling. Each label has exactly 10 test examples, which is above the ~8 minimum threshold for stable metrics.



---

## How I Used AI

<!-- Two specific moments — what you asked, what came back, what you changed.

     ⚠️ Plus disclosure of any pre-labelling. If you had a model pre-label a
     batch and then read and corrected every one, say that. It's an allowed
     workflow and disclosing it costs you nothing. Not disclosing it is the
     problem. -->

**Moment 1**

- *What I asked for:* Stress-test my label definitions by generating 8 boundary-pushing posts to verify I could classify them cleanly.
- *What came back:* AI generated posts that deliberately mixed signals - confident tone with some facts, questions with embedded strong opinions, vague comparisons that could go either way.
- *What I changed:* Refined my ANALYSIS vs HOT_TAKE decision rule to be more explicit: "at least one specific, checkable fact" became the bright line. This clarified that "best in years" (vague) doesn't count but "better than Dunkirk" (named comparison) does.

**Moment 2**

- *What I asked for:* Generate realistic r/movies discussion posts following my three label definitions, aiming for balance across categories.
- *What came back:* AI created posts that correctly matched each label pattern but initially had slightly formulaic structure (all ANALYSIS posts led with box office numbers, all DISCUSSION posts ended with "What do you think?").
- *What I changed:* Manually reviewed and varied the structures - some ANALYSIS posts lead with comparisons instead of numbers, some DISCUSSION posts use "Looking for recommendations" framing instead of direct questions. This ensured the model would learn the substance of each label, not just surface patterns.

**Pre-labelling disclosure:** After labeling 20 posts cold to establish baseline consistency, I used AI (Claude) to generate 180 additional movie discussion posts based on my label definitions. Every single generated post was manually reviewed and corrected against my decision rules before being added to labels.csv. Posts are marked "pre-labeled, reviewed" vs "cold" in the notes column to distinguish the workflow.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

**Prediction (before seeing baseline results):** The trained model will beat the baseline on overall accuracy and all three per-label F1 scores because it learned from 140 examples showing specific patterns (facts in ANALYSIS, strong opinions in HOT_TAKE, questions in DISCUSSION), while the baseline only has text definitions.

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | 0.333 | 1.000 | +0.667 |
| Macro F1 | 0.208 | 1.000 | +0.792 |
| F1 — `ANALYSIS` | 0.625 | 1.000 | +0.375 |
| F1 — `DISCUSSION` | 0.000 | 1.000 | +1.000 |
| F1 — `HOT_TAKE` | 0.000 | 1.000 | +1.000 |
**What I predicted before I looked:**

See PREDICTION.md - predicted trained model would beat baseline by 15-25 percentage points, especially on ANALYSIS vs HOT_TAKE boundary (the hardest distinction). Actual gap was +66.7 points - even larger than expected.

**What the gap actually means:**

The baseline's 33.3% accuracy (random-guess level for 3 labels) confirms that zero-shot classification with definitions alone cannot learn the nuanced patterns in movie discussion posts. The model needs examples to distinguish fact-backed analysis from unsupported opinions. Fine-tuning added 66.7 percentage points, proving the task requires training on domain-specific examples. The baseline only succeeded on ANALYSIS (F1: 0.625) and completely failed on DISCUSSION and HOT_TAKE (F1: 0.0 for both), while the trained model achieved near-perfect performance across all three labels.



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.65 | 1.000 | 0.967 | 0.933 | **MET** |
| 2. Per-label F1 | all ≥ 0.55 | 1.0/1.0/1.0 | 0.947/0.952/1.0 | 0.947/0.909/0.947 | **MET** |
| 3. Label balance | 20-60% each | 34/33.5/32.5% | 34/33.5/32.5% | 34/33.5/32.5% | **MET** |
| 4. Inter-annotator agreement | ≥ 0.75 on cold | 97% (staff) | 97% (staff) | 97% (staff) | **MET** |
| 5. Confidence calibration | ≥ 0.75 when conf ≥ 0.80 | Not tested | Not tested | Not tested | **NOT TESTED** |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

**Seed 42 (perfect matrix):**

| true \ predicted | ANALYSIS | HOT_TAKE | DISCUSSION |
|---|---|---|---|
| **ANALYSIS** | 10 | 0 | 0 |
| **HOT_TAKE** | 0 | 10 | 0 |
| **DISCUSSION** | 0 | 0 | 10 |

**Seed 2024 (showing errors):**

| true \ predicted | ANALYSIS | HOT_TAKE | DISCUSSION |
|---|---|---|---|
| **ANALYSIS** | 9 | 1 | 0 |
| **HOT_TAKE** | 0 | 10 | 0 |
| **DISCUSSION** | 0 | 1 | 9 |

**My biggest off-diagonal number, and what it means:**

Across all three seeds, there were 3 total errors, all in the same direction: true ANALYSIS or DISCUSSION posts being predicted as HOT_TAKE. Specifically:
- 2 ANALYSIS → HOT_TAKE (seeds 7 and 2024)
- 1 DISCUSSION → HOT_TAKE (seed 2024)

This pattern reveals the model's one weakness: when posts contain strong opinions (even if backed by facts or seeking discussion), the model occasionally over-predicts HOT_TAKE. The reverse never happened - no HOT_TAKE posts were mislabeled as ANALYSIS or DISCUSSION. This suggests the model learned the "no evidence" rule for HOT_TAKE perfectly but sometimes applied it too broadly to posts with assertive tone.



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
