"""uaere CLI — eval, twin, demo, swarm, edge. Never NotImplementedError on a supported path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uaere import __version__


def _cap(value: int | None, default: int) -> int:
    return default if value is None else int(value)


def _species_steps(steps: int) -> int:
    """Species logistic budget. Higher than 2000 is kept. The default 250 is not."""
    return max(int(steps), 2000)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="uaere", description="AHAIF underwater acoustic event reasoning")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="run evaluation suites")
    ev.add_argument("--suite", default="paper", choices=["paper", "fish", "species", "review"])
    ev.add_argument("--out", default="artifacts/paper")
    ev.add_argument("--n-windows", type=int, default=240)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--steps", type=int, default=250)
    ev.add_argument("--root", default=None, help="fish WAVs, or one folder per species")
    ev.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="file cap per class. Fish default 3000. Species default 500. Pass a number to override.",
    )
    ev.add_argument("--min-files", type=int, default=40)
    ev.add_argument("--min-groups", type=int, default=8)
    ev.add_argument("--min-classes", type=int, default=10)

    eg = sub.add_parser("evaluate", help="grade a run against frozen EvaluationConditions")
    eg.add_argument("--out", default="artifacts/eval")
    eg.add_argument("--n-windows", type=int, default=None)
    eg.add_argument("--seed", type=int, default=None)
    eg.add_argument("--steps", type=int, default=None)

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

    tr = sub.add_parser("train", help="train a registered backbone on marine sound")
    tr.add_argument("--scenario", default="busy_strait")
    tr.add_argument("--n", type=int, default=200)
    tr.add_argument("--steps", type=int, default=250)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="artifacts/models")
    tr.add_argument("--backbone", default="logistic_mel")
    tr.add_argument("--dataset", default="twin", choices=["twin", "watkins", "fish", "species"])
    tr.add_argument("--root", default=None, help="Watkins, fish, or species root (one folder per class)")
    tr.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="file cap per class. Fish default 3000. Species default 500. Pass a number to override.",
    )
    tr.add_argument("--min-files", type=int, default=40)
    tr.add_argument("--min-groups", type=int, default=8)
    tr.add_argument("--min-classes", type=int, default=10)

    prd = sub.add_parser("predict", help="classify a wav: class, C_wake, WHY/WHY-NOT")
    prd.add_argument("--wav", required=True)
    prd.add_argument(
        "--checkpoint",
        default=None,
        help="species_head.joblib if present, else fish_head.joblib. Will not retrain.",
    )

    kg = sub.add_parser("kg", help="export marine acoustic KG")
    kg.add_argument("--out", default="artifacts/kg/marine_acoustic.ttl")

    dm = sub.add_parser("demo", help="live swarm GUI + Unity JSON twin")
    dm.add_argument("--nodes", type=int, default=8)
    dm.add_argument("--scenario", default="pam_field")
    dm.add_argument(
        "--model",
        default="auto",
        choices=["auto", "oracle", "logistic", "fish", "species"],
        help="auto = species checkpoint, else fish, else ship logistic. oracle is a twin dummy, not a paper result.",
    )
    dm.add_argument("--replay", default=None, help="real fish WAV folder to classify and play")
    dm.add_argument("--checkpoint", default=None)
    dm.add_argument("--seed", type=int, default=0)
    dm.add_argument("--port", type=int, default=8765)
    dm.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")

    sw = sub.add_parser("swarm", help="run N cheap nodes for one tick (no GUI)")
    sw.add_argument("--nodes", type=int, default=8)
    sw.add_argument("--ticks", type=int, default=5)
    sw.add_argument("--scenario", default="pam_field")
    sw.add_argument("--model", default="oracle", choices=["oracle", "logistic", "fish", "species"])
    sw.add_argument("--replay", default=None)
    sw.add_argument("--checkpoint", default=None)
    sw.add_argument("--seed", type=int, default=0)

    ed = sub.add_parser("edge", help="simulate Raspberry Pi field on this machine")
    ed.add_argument("--n", type=int, default=4)
    ed.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)
    if args.cmd == "eval":
        if args.suite == "review":
            from uaere.eval.review_upgrade import run_review_upgrade

            out = args.out if args.out != "artifacts/paper" else "artifacts/paper/review_upgrade"
            print("review suite runs seeds 0, 1, and 2. --seed is not used.", flush=True)
            try:
                report = run_review_upgrade(
                    root=args.root,
                    out_dir=out,
                    train_steps=max(int(args.steps), 2000),
                    max_per_class=_cap(args.max_per_class, 3000),
                    min_files=args.min_files,
                )
            except FileNotFoundError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            summary = report["summary"]["classification_macro_f1"]
            print(json.dumps({"macro_f1_mean": summary["mean"], "macro_f1_std": summary["std"]}, indent=2))
            print(f"wrote {out}/review_upgrade.json")
            return 0
        if args.suite == "fish":
            from uaere.eval.fish_suite import run_fish_suite

            try:
                report = run_fish_suite(
                    root=args.root,
                    out_dir=args.out,
                    seed=args.seed,
                    train_steps=args.steps,
                    max_per_class=_cap(args.max_per_class, 3000),
                    min_files=args.min_files,
                )
            except FileNotFoundError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(json.dumps(report["headline"], indent=2))
            print(f"wrote {args.out}/fish_suite.json")
            return 0
        if args.suite == "species":
            from uaere.data.species import SpeciesDataError
            from uaere.eval.species_suite import run_species_suite

            try:
                report = run_species_suite(
                    root=args.root,
                    out_dir=args.out,
                    seed=args.seed,
                    train_steps=_species_steps(args.steps),
                    max_per_class=_cap(args.max_per_class, 500),
                    min_files=args.min_files,
                    min_groups=args.min_groups,
                    min_classes=args.min_classes,
                )
            except (FileNotFoundError, SpeciesDataError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(json.dumps(report["headline"], indent=2))
            print(f"wrote {args.out}/species_suite.json")
            return 0
        from uaere.eval.suite import run_paper_suite

        report = run_paper_suite(out_dir=args.out, n_windows=args.n_windows, seed=args.seed, train_steps=args.steps)
        print(json.dumps(report["success"], indent=2))
        print(f"wrote {args.out}/suite.json")
        return 0 if all(report["success"].values()) else 1
    if args.cmd == "evaluate":
        from dataclasses import replace

        from uaere.eval.evaluator import EvaluationConditions, Evaluator

        c = EvaluationConditions()
        if args.n_windows is not None:
            c = replace(c, n_windows=args.n_windows)
        if args.seed is not None:
            c = replace(c, seeds=(args.seed,))
        if args.steps is not None:
            c = replace(c, train_steps=args.steps)
        verdict = Evaluator(c).run(out_dir=args.out)
        print(json.dumps({"passed": verdict.passed, "failed": verdict.failed}, indent=2))
        print(f"wrote {args.out}/verdict.md")
        return 0 if verdict.passed else 1
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
        from uaere.data.adapters import WatkinsAdapter
        from uaere.models.logistic_mel import LogisticMelBackbone
        from uaere.models.registry import get_backbone
        from uaere.twin.render import TwinRenderer

        if args.dataset == "fish":
            from uaere.data.adapters import FishCallAdapter
            from uaere.models.fish_head import fit_fish_head, save_fish_checkpoint

            if not args.root:
                print("uaere train --dataset fish requires --root", file=sys.stderr)
                return 2
            try:
                train_recs, _val, _test, meta = FishCallAdapter(
                    args.root,
                    seed=args.seed,
                    max_per_class=_cap(args.max_per_class, 3000),
                    min_files=args.min_files,
                ).load_split()
            except FileNotFoundError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            head = fit_fish_head(train_recs, steps=args.steps, seed=args.seed)
            out = Path(args.out)
            path = save_fish_checkpoint(head, out / "fish_head.joblib", meta=meta)
            print(
                f"saved {path} labels={list(head.labels)} "
                f"n_train={len(train_recs)} split={meta.get('split_unit')}"
            )
            return 0
        if args.dataset == "species":
            from uaere.data.species import SpeciesDataError, load_species_split
            from uaere.models.species_head import save_species_checkpoint, select_species_head

            if not args.root:
                print(
                    "uaere train --dataset species requires --root. "
                    "It does not fall back to the synthetic twin or the toadfish head.",
                    file=sys.stderr,
                )
                return 2
            try:
                train_recs, val_recs, _test, meta = load_species_split(
                    args.root,
                    seed=args.seed,
                    max_per_class=_cap(args.max_per_class, 500),
                    min_files=args.min_files,
                    min_groups=args.min_groups,
                    min_classes=args.min_classes,
                    require_recording_ids=False,
                )
                head, selection = select_species_head(
                    train_recs, val_recs, steps=_species_steps(args.steps), seed=args.seed
                )
            except (FileNotFoundError, SpeciesDataError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            selection.pop("_models", None)
            meta = dict(meta)
            meta["selection"] = selection
            path = save_species_checkpoint(
                head, Path(args.out) / "species_head.joblib", meta=meta, selection=selection
            )
            print(
                f"saved {path} kind={selection['winner']} labels={list(head.labels)} "
                f"val_macro_f1={selection['val_macro_f1']:.4f} "
                f"n_train={len(train_recs)} split={meta.get('split_unit')} "
                f"fs={meta.get('working_fs')}"
            )
            print(meta.get("split_note") or "")
            return 0
        if args.dataset == "watkins":
            recs = WatkinsAdapter(args.root).load()
            if not recs:
                print("no Watkins wavs at --root; falling back to twin")
                recs = TwinRenderer(args.scenario, seed=args.seed).render_dataset(args.n)
        else:
            recs = TwinRenderer(args.scenario, seed=args.seed).render_dataset(args.n)
        bb = get_backbone(args.backbone, seed=args.seed, steps=args.steps)
        if isinstance(bb, LogisticMelBackbone):
            head = bb.fit_records(recs)
        else:
            bb.fit_records(recs)  # type: ignore[attr-defined]
            head = bb.head  # type: ignore[attr-defined]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        npz = out / "last.npz"
        import numpy as np

        np.savez(npz, w1=head.w1, b1=head.b1, w2=head.w2, b2=head.b2, backbone=np.array(args.backbone))
        print(f"saved {npz} backbone={args.backbone} n={len(recs)}")
        return 0
    if args.cmd == "predict":
        from uaere.models.predict import predict_wav

        try:
            report = predict_wav(args.wav, checkpoint=args.checkpoint)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, default=float))
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

        try:
            srv = serve_demo(
                n_nodes=args.nodes,
                scenario=args.scenario,
                seed=args.seed,
                port=args.port,
                model=args.model,
                replay_root=args.replay,
                checkpoint=args.checkpoint,
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"AHAIF twin GUI  → {srv.url}")
        print(f"Unity JSON      → {srv.url}api/state")
        print(f"scenario={args.scenario}  model={srv.state.field.model}")
        print("Click a node and press play to hear the window that node scored.")
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

        field = SwarmField(
            n_nodes=args.nodes,
            scenario=args.scenario,
            seed=args.seed,
            model=args.model,
            replay_root=args.replay,
            checkpoint=args.checkpoint,
        )
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
