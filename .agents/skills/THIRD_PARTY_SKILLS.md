# Third-party skills (vendored 2026-11-09)

| Skill folder | Source | Commit | License |
|---|---|---|---|
| kaggle | https://github.com/shepsci/kaggle-skill (skills/kaggle) | a47bff9 | MIT |
| fiftyone-model-evaluation, fiftyone-find-duplicates, fiftyone-dataset-import, fiftyone-dataset-curation | https://github.com/voxel51/fiftyone-skills (skills/*) | f5c50de | Apache-2.0 |

Notes:
- kaggle: account-changing commands (submit, notebook-run, publish) are dry-run first and need explicit approval. Set KAGGLE_SKILL_READ_ONLY=1 to block them entirely. Daily limit for this competition is 4 submissions / 24h.
- fiftyone-*: need `pip install fiftyone` locally. Used for dataset QA and per-class mAP50 error analysis.
- Not vendored: pjt222/agent-almanac deploy-edge-ai-model (aimed at mobile/LLM deployment, conflicts with TFLM-on-MCU rules).
