# Implementation Tasks: 100% Vietnamese TextCNN Intent Classifier

**Branch**: `002-textcnn-vietnamese-100` | **Spec**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/002-textcnn-vietnamese-100/spec.md) | **Plan**: [plan.md](file:///home/bim/Projects/EAI-2026/specs/002-textcnn-vietnamese-100/plan.md)

---

## Tasks

- [x] **TSK-101**: Build comprehensive Vietnamese corpus generator in `qualifiers/src/intent/augment_corpus.py`:
  - Synthesized 4,485 balanced query patterns for all 5 intents with accented and unaccented pairs.
  - Injected hard negatives for OOS, ambiguous spatial queries, and multi-word object classes.
  - Output saved to `qualifiers/data/intent_train_augmented.csv`.
- [x] **TSK-102**: Upgrade text preprocessing in `qualifiers/src/intent/text_cnn.py`:
  - Added canonical NFD-based Vietnamese accent normalizer (`remove_accents`).
  - Built comprehensive, compact vocabulary mapping in `qualifiers/weights/text_cnn_vocab.json` (206 canonical tokens).
- [x] **TSK-103**: Retrain TextCNN model and verify:
  - 100% accuracy on `intent_val.csv` (158/158).
  - 100% accuracy on `intent_test.csv` (156/156).
  - 100% accuracy on extended 50-sample real-world stress test suite (50/50).
- [x] **TSK-104**: Export and quantize to Full Integer INT8 TFLite Micro (`text_cnn_int8.tflite`):
  - Model size verified: 19.36 KB (well below 1,500 KB limit).
  - Total parameters: 11,493 parameters (strictly $< 200,000$).
  - Zero accuracy degradation: TFLite evaluation achieves 100.00% on both val and test sets.
- [x] **TSK-105**: Update `qualifiers/submissions/intent_preds.csv` and verify with `qualifiers/submit.py`:
  - 158 valid predictions saved, validated against ground truth (100.00%).

---

## Convergence Status: CONVERGED

Model training, quantization, stress testing, and submission verification completed with 100.00% accuracy across all Vietnamese evaluation sets.
