#!/usr/bin/env python
"""End-to-end paper suite. Thin wrapper around uaere.eval.suite."""

from uaere.eval.suite import run_paper_suite

if __name__ == "__main__":
    r = run_paper_suite()
    print(r["success"])
