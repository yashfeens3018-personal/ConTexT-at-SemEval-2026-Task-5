# ConTexT at SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Stories through Narrative Understanding

This repository contains the data, code and results for our system submitted to **SemEval-2026 Task 5** (Gehring et al., 2026). The paper was published at the 20th International Workshop on Semantic Evaluation (SemEval-2026), San Diego, California, USA.

📄 **Paper:** [ACL Anthology](https://aclanthology.org/2026.semeval-1.232/) | [PDF](https://aclanthology.org/2026.semeval-1.232.pdf)
🔗 **DOI:** [10.18653/v1/2026.semeval-1.232](https://doi.org/10.18653/v1/2026.semeval-1.232)

**Authors:** Fakeha Faisal*, Syeda Zaidi*, Rubab Shah*, Azkaa Nasir*, Sandesh Kumar, Abdul Samad
Department of Computer Science, Habib University, Karachi, Pakistan
\* Equal contribution

---

## About the task

Traditional word sense disambiguation (WSD) picks one correct sense for a word. In real narratives though, interpretation is subjective, and different people give different degrees of plausibility to the same sense.

SemEval-2026 Task 5 reframes WSD as a **graded prediction task**. Given a short story containing an ambiguous homonym and a candidate meaning, the system has to predict a plausibility score from **1 to 5**, which reflects the average judgment of five independent human annotators instead of a single gold label.

Systems are evaluated with:
- **Spearman correlation (ρ)** between predicted and human plausibility ratings
- **Accuracy within standard deviation (Acc)**: the share of predictions that fall within one standard deviation of the mean human rating

## Our results

Our best system is **DeBERTa-xLarge with a three-stage curriculum** (WiC pretraining → Wasserstein distribution learning → KL calibration).

| Metric | Score |
|---|---|
| Accuracy within 1 SD | **78%** |
| Spearman ρ | **0.70** |
| Average (overall) score | **0.74** |

This ranked **31st out of 79** participating teams. The main takeaway from our experiments is that modeling the full distribution of human ratings matches human plausibility judgments better than predicting a single score.

---

## Repository structure

```
.
├── AmbiStory/                     # Task data (train / validation / test) and related files
├── WiC/                           # Word-in-Context data used for pretraining
├── gwsd/                          # Graded Word Sense Disambiguation data used for augmentation
├── scoring/                       # Scoring / evaluation scripts
├── 2026.semeval-1.232.pdf         # Our system paper
└── README.md
```

---

## Data

**AmbiStory (SemEval-2026 Task 5)**
- Train: **2,280** entries
- Validation: **588** entries
- Test: **930** entries

Each entry has:
- `homonym`: the ambiguous word
- `judged_meaning`: the candidate sense being rated
- `precontext`, `sentence`, `ending`: the three parts of the short story (the sentence contains the target homonym, the ending is optional)
- `average`: mean of the five human ratings (the gold plausibility score)
- `stdev`: standard deviation of the ratings (annotator agreement)
- `example_sentence`: a sentence showing a clear usage of the homonym in its judged meaning

**WiC (Word-in-Context):** a binary task where the model decides whether a target word has the same meaning in two contexts (Pilehvar and Camacho-Collados, 2019). We used it for pretraining.

**GWSD (Graded Word Sense Disambiguation):** from Cassotti and Tahmasebi (2025). Its structure is close to SemEval's, so we used it to augment our data. Its scale is 0-4 instead of 1-5, so we mapped it like this:
- a score of 0 (annotator uncertainty) → 3, the neutral midpoint
- 1→1, 2→2, 3→4, 4→5

GWSD has no example sentences, so we generated them with Gemini Flash 2.5. These were **not manually reviewed**, so they can add some noise.

**Agreement-aware grouping.** We split instances by mean plausibility (μ) and standard deviation (σ):

| Group | Description | Condition |
|---|---|---|
| G1 | low mean, low variance | σ ≤ 1.5, μ ≤ 3 |
| G2 | high mean, low variance | σ ≤ 1.5, μ > 3 |
| G3 | high variance | σ > 1.5 |

---

## Method

**Input.** The full story (precontext + target sentence with `[TGT]` markers + ending) and the judged meaning (definition + example sentence) are jointly encoded with a cross-encoder.

**Phase 1: Embedding similarity.** Cosine similarity between story and meaning embeddings, then target word and meaning embeddings, with GWSD augmentation, ensemble training (bagging) and text augmentation (random insertion/deletion, back-translation, WordNet synonyms, hypernym concatenation, synthetic gloss and example generation).

**Phase 2: Stronger representation and supervision.** Example-sentence anchored comparison, WiC transfer learning, model scaling (BERT, RoBERTa, ELECTRA, MPNet, DeBERTa-large, DeBERTa-xLarge) and variance-aware loss experiments.

**Phase 3: Distribution matching.** Instead of Gaussian-smoothed soft labels (as in SORD), we normalize the empirical human rating distribution directly, so the real annotator disagreement is kept.

**Three-stage curriculum (final system):**

| Stage | What we do |
|---|---|
| 1. WiC pretraining | Target token marking, sentence-pair cross-encoding, partial layer freezing (only upper layers updated), cross-entropy loss |
| 2. Wasserstein distribution learning | Predict a distribution over the 5 ordinal classes with the loss `0.8 · EMD + 0.2 · (E[p] − E[p̂])²` |
| 3. High-agreement KL calibration | Fine-tune on the G1 and G2 subsets with `0.7 · KL + 0.3 · (E[p] − E[p̂])²` |

**Training setup:** batch size 16, linear learning rate scheduler with warmup, early stopping on dev loss. Learning rate is 1e-5 for WiC pretraining and distribution learning, and 5e-6 for KL calibration.

**Output:** a probability distribution over the five plausibility classes. The expected value is used as the continuous score (for Spearman) and the rounded value for discrete evaluation.

---

## Results

| Phase | Models | Approach | Acc (%) | ρ |
|---|---|---|---|---|
| 1 | BERT-base, RoBERTa-base | Story vs. meaning similarity | 44 | 0.33 |
| 2 | BERT-base, RoBERTa-base | Target word vs. meaning similarity | 51 | 0.36 |
| 3 | BERT-base, RoBERTa-base | Same as 2 + GWSD (G1) | 58 | 0.42 |
| 4 | BERT-base, RoBERTa-base, DistilBERT, DeBERTa-v3-base, ELECTRA-small | Same as 3 + ensemble training | 58 | 0.43 |
| 5 | BERT-base, RoBERTa-base, DeBERTa-large | Same as 2 + random insertion/deletion | 58 | 0.42 |
| 6 | MPNet-base-v2, DeBERTa-large | Example sentence anchoring + back-translation (German) + LR scheduler | 69 | 0.46 |
| 7a | DeBERTa-large | SemEval only | 66 | 0.62 |
| 7b | DeBERTa-large | SemEval + GWSD (G1+G2) | 68 | 0.66 |
| 7c | DeBERTa-large | SemEval + GWSD (G1+G2+G3) | 69 | 0.67 |
| 7d | DeBERTa-large | Gloss back-translation (G1,G2) + SemEval | 65 | 0.59 |
| 7e | DeBERTa-large | Hypernym concatenation + SemEval | 75 | 0.60 |
| 7f | DeBERTa-large | WiC → SemEval pretraining | 69 | 0.68 |
| 7g | DeBERTa-large | WiC → SemEval + GWSD (G1+G2+G3) | 68 | 0.69 |
| 8a | DeBERTa-large | Soft-label distribution learning (KL + MSE) + GWSD | 77 | 0.66 |
| 8b | DeBERTa-xlarge | SemEval + GWSD (G1+G2+G3) | 69 | 0.74 |
| 8c | DeBERTa-large | Wasserstein (EMD) loss | 80 | 0.65 |
| 8d | DeBERTa-large | Variance-weighted loss scaling (σ < 1.5) | 75 | 0.65 |
| **8e** | **DeBERTa-xlarge** | **Full 3-stage curriculum (WiC + EMD hybrid + KL calibration)** | **78** | **0.70** |

**What we saw:**
- Embedding similarity stalled at 58% accuracy, no matter how much data or how many models we added. Cosine similarity gives a single score with no ordinal structure, and it can't capture the spread of human opinion.
- Anchoring to the example sentence (Phase 6) broke that plateau, and full cross-encoder fine-tuning raised ρ a lot.
- WiC pretraining gave the strongest single boost (ρ = 0.68).
- Moving to distribution modeling (EMD loss and KL calibration) matched the evaluation metrics better, and the full curriculum gave the best overall result.

**Error analysis (best model, per group):**

| Group | Acc (%) | ρ |
|---|---|---|
| G1 (low mean, low var.) | 68.0 | 0.56 |
| G2 (high mean, low var.) | 82.0 | 0.36 |
| G3 (high variance) | 93.0 | 0.46 |

G3 has the highest accuracy, probably because high-variance items have higher mean scores that are easier to predict, but the model struggles to rank within that ambiguous middle. G1 is the opposite: lowest accuracy but highest correlation. G2 has the weakest correlation. Cases where the story context differs strongly from the example sentence are harder, and some examples are just genuinely ambiguous.

> Note: results are from single runs (no multiple seeds), so differences of 0.01-0.02 between nearby configurations should be seen as trends and not significant gains.

---

## Limitations

- High computational cost and limited scalability (large encoders and multi-stage training).
- Evaluated only on English narratives.
- Loss mixing coefficients were tuned on the validation set without sensitivity analysis.
- The distributional approach assumes annotator homogeneity and doesn't model individual annotator bias.
- No confidence intervals or significance testing across random seeds.
- Gains from the curriculum are smaller for highly context-dependent homonyms.

## Future work

- Uncertainty-aware or annotator-aware modeling to capture structured disagreement.
- Parameter-efficient fine-tuning of instruction-tuned LLMs with distribution-aware objectives.
- Extending to multilingual and low-resource settings.
- Validating design choices across multiple seeds.

---

## Getting started

```bash
git clone https://github.com/yashfeens3018-personal/ConTexT-at-SemEval-2026-Task-5.git
cd ConTexT-at-SemEval-2026-Task-5
```

## References

- Gehring, Meyer and Roth. 2026. *SemEval-2026 Task 5: Rating plausibility of word senses in ambiguous stories through narrative understanding.*
- Pilehvar and Camacho-Collados. 2019. *WiC: the Word-in-Context dataset.* NAACL-HLT.
- Cassotti and Tahmasebi. 2025. *Sense-specific historical word usage generation.* TACL.
- Yap, Koh and Chng. 2020. *Adapting BERT for word sense disambiguation with gloss selection objective and example sentences.* Findings of EMNLP.
- Díaz and Marathe. 2019. *Soft labels for ordinal regression.* CVPR.
- Hou, Yu and Samaras. 2016. *Squared earth mover's distance-based loss for training deep neural networks.*
