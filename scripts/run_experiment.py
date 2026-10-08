#!/usr/bin/env python
"""Thin wrapper so `python scripts/run_experiment.py ...` works after `pip install -e .`."""
from mtkd_adr.run import main

if __name__ == "__main__":
    main()
