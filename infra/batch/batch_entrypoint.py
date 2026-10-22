#!/usr/bin/env python3
import os

from manuscript_analysis.cloud import ablation, null


if __name__ == "__main__":
    if "ABLATION_MODE" in os.environ:
        ablation.run_worker_from_env()
    else:
        null.run_worker_from_env()
