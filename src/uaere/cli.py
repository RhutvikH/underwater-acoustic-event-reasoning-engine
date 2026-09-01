"""uaere CLI — eval, twin, demo, swarm, edge. Never NotImplementedError on a supported path."""

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

    dm = sub.add_parser("demo", help="live swarm GUI + Unity JSON twin")
    dm.add_argument("--nodes", type=int, default=8)
    dm.add_argument("--scenario", default="busy_strait")
    dm.add_argument("--seed", type=int, default=0)
    dm.add_argument("--port", type=int, default=8765)
    dm.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")

    sw = sub.add_parser("swarm", help="run N cheap nodes for one tick (no GUI)")
    sw.add_argument("--nodes", type=int, default=8)
    sw.add_argument("--ticks", type=int, default=5)
    sw.add_argument("--scenario", default="busy_strait")
    sw.add_argument("--seed", type=int, default=0)

    ed = sub.add_parser("edge", help="simulate Raspberry Pi field on this machine")
    ed.add_argument("--n", type=int, default=4)
    ed.add_argument("--seed", type=int, default=0)

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
    if args.cmd == "demo":
        from uaere.demo.server import serve_demo
        import time

        srv = serve_demo(n_nodes=args.nodes, scenario=args.scenario, seed=args.seed, port=args.port)
        print(f"AHAIF twin GUI  → {srv.url}")
        print(f"Unity JSON      → {srv.url}api/state")
        print("Ctrl-C to stop.")
        try:
            if args.seconds > 0:
                time.sleep(args.seconds)
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        srv.stop()
        return 0
    if args.cmd == "swarm":
        from uaere.swarm.field import SwarmField, tick_to_json

        field = SwarmField(n_nodes=args.nodes, scenario=args.scenario, seed=args.seed)
        field.warmup()
        last = None
        for _ in range(args.ticks):
            last = field.step()
        print(json.dumps(tick_to_json(last), indent=2, default=float))
        return 0
    if args.cmd == "edge":
        from uaere.classify.train import extract_features, train_evidential
        from uaere.edge.runtime import simulate_pi_field
        from uaere.kg.graph import load_kg
        from uaere.pipeline import AhaifPipeline
        from uaere.representation.env_norm import AdaptiveNormalizer
        from uaere.trust.trust_score import TrustEngine
        from uaere.twin.render import TwinRenderer
        from uaere.types import PolicyVector

        recs = TwinRenderer("busy_strait", seed=args.seed).render_dataset(max(args.n, 16))
        head = train_evidential(extract_features(recs), steps=80, seed=args.seed)
        pipe = AhaifPipeline(
            trust=TrustEngine(head),
            kg=load_kg(),
            policy=PolicyVector(0.2, 0.4, 0.65, device_id="raspberry_pi_zero2", require_authenticated=False),
            normalizer=AdaptiveNormalizer(32),
        )
        out = simulate_pi_field(pipe, recs, n=args.n)
        print(json.dumps(out, indent=2, default=float))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
