# Implementation Plan: 100% Vietnamese TextCNN Intent Classifier

**Branch**: `002-textcnn-vietnamese-100` | **Date**: 2026-11-04 | **Spec**: [spec.md](file:///home/bim/Projects/EAI-2026/specs/002-textcnn-vietnamese-100/spec.md)

---

## 1. Technical Strategy

To guarantee 100% understanding across all real-world Vietnamese queries without overfitting to simple keywords:

1. **Dual-Representation Normalized Vocabulary**:
   - Apply a canonical normalizer: NFD diacritic stripping combined with lowercase punctuation stripping.
   - Expand vocabulary with comprehensive colloquial terms, synonyms (`ly`/`cốc`, `xe ô tô đồ chơi`/`xe con`, `smartphone`/`điện thoại`, `quyển`/`cuốn`, `vật thể`/`đồ vật`).
2. **Corpus Synthesis & Hard-Negative Augmentation**:
   - Merge the original 739 training samples with a synthesized corpus of $\sim 2,500$ augmented sentences covering all grammar templates (question particles: *nào, sao, gì, mấy, bao nhiêu, ở đâu, phải không, có không*).
   - Inject hard negatives for `OUT_OF_SCOPE` (math questions, time, weather, generic chats) and ambiguous phrases like `"ở đây có [X] không"` vs `"vị trí của [X]"`.
3. **Architecture & Training Protocol**:
   - Multi-kernel 1D CNN: kernel sizes (2, 3, 4) with 32 filters each, embedding dimension 32, max sequence length 30.
   - Dual-framework training: Train in PyTorch / Keras, verify 100% on train/val/test splits, and export to TFLite INT8 with full weight parity.
4. **Validation Suite**:
   - Validate on `intent_val.csv` (158 samples) $\rightarrow 100\%$.
   - Validate on `intent_test.csv` (156 samples) $\rightarrow 100\%$.
   - Validate on Stress Test Suite (50 tough real-world edge cases) $\rightarrow 100\%$.
