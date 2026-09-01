"""uaere train|eval|twin|data — never NotImplementedError on the CLI path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uaere import __version__


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="uaere", description="AHAIF underwater acoustic event reasoning")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="run evaluation suites")
    ev.add_argument("--suite", default="paper", choices=["paper"])
    ev.add_argument("--out", default="artifacts/paper")
    ev.add_argument("--n-windows", type=int, default=240)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--steps", type=int, default=250)

    tw = sub.add_parser("twin", help="render twin windows")
    tw.add_argument("--scenario", default="busy_strait")
    tw.add_argument("--n", type=int, default=32)
    tw.add_argument("--seed", type=int, default=0)
    tw.add_argument("--out", default="artifacts/twin")

    ds = sub.add_parser("data", help="dataset summaries")
    ds.add_argument("action", choices=["summarize"])
    ds.add_argument("--scenario", default="busy_strait")
    ds.add_argument("--n", type=int, default=64)
    ds.add_argument("--seed", type=int, default=0)

    tr = sub.add_parser("train", help="train evidential head on twin-synthetic")
    tr.add_argument("--scenario", default="busy_strait")
    tr.add_argument("--n", type=int, default=200)
    tr.add_argument("--steps", type=int, default=250)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="artifacts/models")

    kg = sub.add_parser("kg", help="export marine acoustic KG")
    kg.add_argument("--out", default="artifacts/kg/marine_acoustic.ttl")

    args = p.parse_args(argv)
    if args.cmd == "eval":
        from uaere.eval.suite import run_paper_suite

        report = run_paper_suite(out_dir=args.out, n_windows=args.n_windows, seed=args.seed, train_steps=args.steps)
        print(json.dumps(report["success"], indent=2))
        print(f"wrote {args.out}/suite.json")
        return 0 if all(report["success"].values()) else 1
    if args.cmd == "twin":
        from uaere.data.wavutil import write_wav
        from uaere.twin.render import TwinRenderer

        recs = TwinRenderer(args.scenario, seed=args.seed).render_dataset(args.n)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for r in recs:
            write_wav(out / f"{r.source_id}.wav", r.waveform, r.sample_rate)
        print(f"rendered {len(recs)} windows → {out}")
        return 0
    if args.cmd == "data":
        from uaere.data.adapters import TwinReplayAdapter

        ad = TwinReplayAdapter(args.scenario, seed=args.seed, n_windows=args.n)
        print(json.dumps(ad.summarize(), indent=2))
        return 0
    if args.cmd == "train":
        from uaere.classify.train import extract_features, train_evidential
        from uaere.twin.render import TwinRenderer

        recs = TwinRenderer(args.scenario, seed=args.seed).render_dataset(args.n)
        bun = extract_features(recs)
        head = train_evidential(bun, steps=args.steps, seed=args.seed)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        npz = out / "evidential_head.npz"
        import numpy as np

        np.savez(npz, w1=head.w1, b1=head.b1, w2=head.w2, b2=head.b2)
        print(f"saved {npz}")
        return 0
    if args.cmd == "kg":
        from uaere.kg.graph import load_kg

        g = load_kg()
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(g.to_turtle(), encoding="utf-8")
        print(f"wrote {path} ({g.n_nodes} nodes, {g.n_edges} edges)")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
