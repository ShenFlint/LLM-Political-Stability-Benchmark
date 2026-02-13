# Replication Materials for "Before You Simulate"

This repository contains the dataset, persona profiles, and analysis code for the paper:

**Before You Simulate: A Pre-Study Benchmark for Large Language Model Stability in Political Role-Playing Simulations**

## Repository Structure

```text
├── persona_profiles.jsonl        # The 15 anonymized persona profiles used in the study
├── political_result_1/           # Raw response data (62 items x 5 runs x conditions)
├── score/                        # Calculated Political Compass coordinate scores
├── lmm_icc/                      # Output logs/results from Linear Mixed-Effects Models
├── summary_oss.csv               # Aggregated Overall Stability Score (OSS) data
├── summary_entropy.csv           # Aggregated Response Entropy data
├── summary_oss_entropy_merged.csv # Merged dataset for statistical analysis
├── get_oss.py                    # Script to compute OSS from raw data
├── get_entropy.py                # Script to compute Response Entropy from raw data
├── get_merge.py                  # Utility script to merge OSS and Entropy datasets
└── run_lmm.py                    # Script to run Linear Mixed-Effects Models (LMM)
