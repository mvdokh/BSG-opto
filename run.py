#!/usr/bin/env python3
"""Entry point for running any experiment in the BSG-opto suite.

Examples:
    .venv/bin/python run.py list
    .venv/bin/python run.py random-exc --pct 15
    .venv/bin/python run.py tonic-altconn --i-max 20 40 80
    .venv/bin/python run.py state-dependence --sweep-only
"""
import sys
from bsg_opto.cli import main

if __name__ == '__main__':
    sys.exit(main())
