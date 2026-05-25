from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import fast_dis_freq_matrix_threaded
from src.metrics import best_corner_local_gaussian
from src.model import inference_model, load_model
from src.plot import plot_acceleration, plot_dis_freq_matrix, plot_prediction

SAMPLE_RECORD = PROJECT_ROOT / "data" / "20260106101800s.pickle"
DEFAULT_EVENT_ROOT = Path(
    os.environ.get("SMR_DP_EVENT_ROOT", "/home/italoif/data/knt_kik/events")
)
DEFAULT_METADATA_CSV = Path(
    os.environ.get("SMR_DP_METADATA_CSV", "/home/italoif/gmr-ml/data/jp_dataset_2026v2.csv")
)


def make_fc_hp_grid(step=0.005, n_values=256):
    return [step * i for i in range(1, n_values + 1)]


def _require_path(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _prediction_summary(xy, acc, dt, time_res, fc_hps):
    row = int(np.clip(round(xy[0]), 0, len(fc_hps) - 1))
    p_arrival = xy[1] / time_res * len(acc) * dt
    return {
        "matrix_y": float(xy[0]),
        "matrix_x": float(xy[1]),
        "p_arrival_s": float(p_arrival),
        "fc_hp_hz": float(fc_hps[row]),
    }


def run_demo(
    record_path=SAMPLE_RECORD,
    trace_index=3,
    fc_lp=30.0,
    time_res=1024,
    max_workers=2,
    sigma=5,
    save_figures=False,
    output_dir=PROJECT_ROOT / "img",
):
    record_path = _require_path(record_path, "Sample record")
    output_dir = Path(output_dir)

    st = read(str(record_path))
    if trace_index < 0 or trace_index >= len(st):
        raise IndexError(f"trace_index={trace_index} is outside stream length {len(st)}")

    tr = st[trace_index]
    acc = np.asarray(tr.data)
    dt = float(tr.stats.delta)
    fc_hps = make_fc_hp_grid()

    t0 = time.perf_counter()
    dfm = fast_dis_freq_matrix_threaded(
        acc,
        dt,
        fc_hps,
        fc_lp=fc_lp,
        time_res=time_res,
        max_workers=max_workers,
    )
    t1 = time.perf_counter()

    pred = inference_model(dfm)
    t2 = time.perf_counter()

    xy = best_corner_local_gaussian(pred, sigma=sigma)
    t3 = time.perf_counter()

    summary = {
        "record_path": str(record_path),
        "trace_id": getattr(tr, "id", f"trace_{trace_index}"),
        "trace_index": trace_index,
        "n_samples": len(acc),
        "dt": dt,
        "matrix_shape": tuple(dfm.shape),
        "dfm_time_s": t1 - t0,
        "inference_time_s": t2 - t1,
        "post_time_s": t3 - t2,
        "total_time_s": t3 - t0,
        **_prediction_summary(xy, acc, dt, time_res, fc_hps),
    }

    if save_figures:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_dis_freq_matrix(dfm, save_path=str(output_dir / "dis_freq_matrix.png"))
        plot_prediction(pred, xy, save_path=str(output_dir / "prediction.png"))
        plot_acceleration(
            acc,
            dt,
            p_arrival=summary["p_arrival_s"],
            save_path=str(output_dir / "acceleration_plot.png"),
        )

    print("=== SMR-DP demo ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


def test_making_matrix():
    """Backward-compatible helper for quick local checks."""
    return run_demo(save_figures=True)


def test_check_time_all_traces(
    event_id="20251125180103",
    filepath=None,
    event_root=DEFAULT_EVENT_ROOT,
    metadata_csv=DEFAULT_METADATA_CSV,
    location="00",
    fc_lp=30.0,
    time_res=1024,
    max_workers=2,
    sigma=5,
    verbose=True,
):
    event_root = Path(event_root)
    metadata_csv = _require_path(metadata_csv, "Metadata CSV")
    filepath = Path(filepath) if filepath else event_root / event_id / f"{event_id}.pickle"
    filepath = _require_path(filepath, "Event waveform file")

    st = read(str(filepath))
    df = pd.read_csv(metadata_csv)

    if "id" not in df.columns:
        raise ValueError(f"Metadata CSV must contain an 'id' column: {metadata_csv}")

    id_parts = df["id"].astype(str).str.split("_", expand=True)
    if id_parts.shape[1] < 4:
        raise ValueError("Expected metadata id format: <event>_<network>_<station>_<channel>")

    df = df.assign(event=id_parts[0], station=id_parts[2], channel=id_parts[3])
    df = df[df["event"] == event_id]
    stations_in_df = set(df["station"].dropna().unique())
    if not stations_in_df:
        raise ValueError(f"No stations found for event_id={event_id} in {metadata_csv}")

    st = st.__class__([tr for tr in st if tr.stats.station in stations_in_df])
    if location:
        st = st.select(location=location)

    print()
    print(f"{len(stations_in_df)} stations in metadata for event {event_id}")
    print(st)

    fc_hps = make_fc_hp_grid()
    model = load_model()
    results = []
    t_all_0 = time.perf_counter()

    for i, tr in enumerate(st):
        acc = np.asarray(tr.data)
        dt = float(tr.stats.delta)

        t0 = time.perf_counter()
        dfm = fast_dis_freq_matrix_threaded(
            acc,
            dt,
            fc_hps,
            fc_lp=fc_lp,
            time_res=time_res,
            max_workers=max_workers,
        )
        t1 = time.perf_counter()

        pred = inference_model(dfm, model=model)
        t2 = time.perf_counter()

        xy = best_corner_local_gaussian(pred, sigma=sigma)
        t3 = time.perf_counter()

        result = {
            "trace_index": i,
            "trace_id": getattr(tr, "id", f"trace_{i}"),
            "n_samples": len(acc),
            "dt": dt,
            "dfm_time_s": t1 - t0,
            "inference_time_s": t2 - t1,
            "post_time_s": t3 - t2,
            "total_time_s": t3 - t0,
            **_prediction_summary(xy, acc, dt, time_res, fc_hps),
        }
        results.append(result)

        if verbose:
            print(
                f"[{i:03d}] {result['trace_id']} | "
                f"dfm={result['dfm_time_s']:.4f}s | "
                f"infer={result['inference_time_s']:.4f}s | "
                f"post={result['post_time_s']:.4f}s | "
                f"total={result['total_time_s']:.4f}s | "
                f"tP={result['p_arrival_s']:.3f}s | "
                f"fcHP={result['fc_hp_hz']:.3f}Hz"
            )

    wall_time = time.perf_counter() - t_all_0
    summary = _timing_summary(results, wall_time)

    print("\n=== Timing summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return results, summary


def _timing_summary(results, wall_time):
    def stats(name):
        values = np.array([r[name] for r in results], dtype=float)
        if len(values) == 0:
            return {f"{name}_min": None, f"{name}_max": None, f"{name}_avg": None}
        return {
            f"{name}_min": float(values.min()),
            f"{name}_max": float(values.max()),
            f"{name}_avg": float(values.mean()),
        }

    summary = {
        "n_traces": len(results),
        "wall_time_s": wall_time,
    }
    for name in ("total_time_s", "dfm_time_s", "inference_time_s", "post_time_s"):
        summary.update(stats(name))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SMR-DP demo inference or local event timing benchmarks."
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="Run inference on the bundled sample record.")
    demo.add_argument("--record-path", default=str(SAMPLE_RECORD))
    demo.add_argument("--trace-index", type=int, default=3)
    demo.add_argument("--fc-lp", type=float, default=30.0)
    demo.add_argument("--time-res", type=int, default=1024)
    demo.add_argument("--max-workers", type=int, default=2)
    demo.add_argument("--sigma", type=float, default=5)
    demo.add_argument("--save-figures", action="store_true")
    demo.add_argument("--output-dir", default=str(PROJECT_ROOT / "img"))

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run timing on all matching traces for a local event file.",
    )
    benchmark.add_argument("--event-id", default="20251125180103")
    benchmark.add_argument("--filepath", default=None)
    benchmark.add_argument("--event-root", default=str(DEFAULT_EVENT_ROOT))
    benchmark.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    benchmark.add_argument("--location", default="00")
    benchmark.add_argument("--fc-lp", type=float, default=30.0)
    benchmark.add_argument("--time-res", type=int, default=1024)
    benchmark.add_argument("--max-workers", type=int, default=2)
    benchmark.add_argument("--sigma", type=float, default=5)
    benchmark.add_argument("--quiet", action="store_true")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "demo"):
        return run_demo(
            record_path=args.record_path if args.command else SAMPLE_RECORD,
            trace_index=args.trace_index if args.command else 3,
            fc_lp=args.fc_lp if args.command else 30.0,
            time_res=args.time_res if args.command else 1024,
            max_workers=args.max_workers if args.command else 2,
            sigma=args.sigma if args.command else 5,
            save_figures=args.save_figures if args.command else False,
            output_dir=args.output_dir if args.command else PROJECT_ROOT / "img",
        )

    if args.command == "benchmark":
        return test_check_time_all_traces(
            event_id=args.event_id,
            filepath=args.filepath,
            event_root=args.event_root,
            metadata_csv=args.metadata_csv,
            location=args.location,
            fc_lp=args.fc_lp,
            time_res=args.time_res,
            max_workers=args.max_workers,
            sigma=args.sigma,
            verbose=not args.quiet,
        )

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
