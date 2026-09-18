"""
Real-hardware inference benchmark for GF-IDS.

Meant to run ON the Raspberry Pi 4 (or any CPU-only machine) -- NOT on
the training GPU. It answers a different question than evaluate.py's
InfTime(ms): that number is wall-clock on the RTX 4060 over a
batch-1024 slice; this script times single-sample (batch=1) inference
on the actual constrained device the paper's deployment story is
about.

Needs only torch, numpy, and this repo's models.py / binary_ops.py /
quant_ops.py -- deliberately no sklearn, no thop, no GPU, so it runs
on a bare Pi image without extra setup.

Files to copy onto the Pi, all into one folder:
    code/pi_benchmark.py          (this file)
    code/models.py
    code/binary_ops.py
    code/quant_ops.py
    code/pi_bench_sample.npz      (500 real test rows, not the full
                                    277 MB X_test.npy -- this script
                                    only measures latency, not accuracy)
    runs/{model}_seed42_best.pt   for each model in --models (the same
                                    checkpoint Table V/VII already
                                    report numbers for)

Then, on the Pi:
    python3 pi_benchmark.py

Optional real energy measurement: if an INA219 or INA226 current-sense
sensor is wired into the Pi's power rail and
`adafruit-circuitpython-ina219` (or `-ina226`) is installed, pass
--energy to sample it during each timed run. Without the flag, or
without the hardware/library present, energy is silently skipped and
only latency is reported -- latency alone already satisfies the
paper's stated minimum bar (measured latency on real hardware); energy
is a bonus once a sensor is wired up.
"""

import argparse
import csv
import platform
import time

import numpy as np
import torch

from models import MODEL_REGISTRY

HEADLINE_MODELS = ["BNN-MATCHED", "MLP", "LSTM", "CNN", "BNN-INT8IO",
                   "MLP-INT8"]


def run_tag(model_name, seed=42):
    """Mirrors federated_train.run_tag() for the plain (IID, not
    class-weighted) runs used in the paper's main tables. Duplicated
    here rather than imported so this script has no dependency on
    federated_train.py's sklearn import, which a bare Pi image may not
    have installed."""
    return f"{model_name}_seed{seed}"


def read_ina219():
    """Return (bus_voltage_V, current_mA), or None if the sensor
    hardware/library isn't available. Safe to call with nothing wired
    up -- just means energy is skipped."""
    try:
        import board
        import busio
        from adafruit_ina219 import INA219
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = INA219(i2c)
        return sensor.bus_voltage, sensor.current
    except Exception:
        return None


def find_checkpoint(name):
    for candidate in (f"../runs/{run_tag(name)}_best.pt",
                      f"runs/{run_tag(name)}_best.pt",
                      f"{run_tag(name)}_best.pt"):
        try:
            return torch.load(candidate, map_location="cpu")
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"{run_tag(name)}_best.pt not found in ../runs/, runs/, or "
        f"the current directory")


def benchmark_model(name, X, input_dim, num_classes, n_warmup, n_iters,
                    n_runs, measure_energy):
    state = find_checkpoint(name)
    model = MODEL_REGISTRY[name](input_dim, num_classes)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        for i in range(n_warmup):
            x = torch.FloatTensor(X[i % len(X):i % len(X) + 1])
            model(x)

    run_means_ms, power_mw = [], []
    with torch.no_grad():
        for _ in range(n_runs):
            before = read_ina219() if measure_energy else None
            start = time.perf_counter()
            for i in range(n_iters):
                x = torch.FloatTensor(X[i % len(X):i % len(X) + 1])
                model(x)
            elapsed = time.perf_counter() - start
            after = read_ina219() if measure_energy else None
            run_means_ms.append(elapsed / n_iters * 1000)
            if before is not None and after is not None:
                v = (before[0] + after[0]) / 2
                i_ma = (before[1] + after[1]) / 2
                power_mw.append(v * i_ma)

    return {
        "Model": name,
        "MeanLatency_ms": round(float(np.mean(run_means_ms)), 4),
        "StdLatency_ms": round(float(np.std(run_means_ms)), 4),
        "N_runs": n_runs,
        "N_iters_per_run": n_iters,
        "MeanPower_mW": (round(float(np.mean(power_mw)), 2)
                         if power_mw else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=HEADLINE_MODELS)
    parser.add_argument("--n-warmup", type=int, default=20,
                        help="untimed inferences before measurement, "
                             "to skip past cold-start effects")
    parser.add_argument("--n-iters", type=int, default=200,
                        help="single-sample inferences timed per run")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="repeated timing runs, for mean +/- std")
    parser.add_argument("--data", default="pi_bench_sample.npz")
    parser.add_argument("--energy", action="store_true",
                        help="also sample an INA219/INA226 over I2C "
                             "during each run (needs the sensor wired "
                             "up and adafruit-circuitpython-ina219 "
                             "installed; silently skipped otherwise)")
    parser.add_argument("--out", default="pi_results.csv")
    args = parser.parse_args()

    data = np.load(args.data)
    X, y = data["X"], data["y"]
    input_dim = X.shape[1]
    num_classes = int(y.max()) + 1

    print(f"Platform: {platform.platform()}, machine={platform.machine()}")
    print(f"torch {torch.__version__}, {len(X)} sample(s) from {args.data}")
    if args.energy and read_ina219() is None:
        print("--energy requested but no INA219/INA226 detected "
              "(check wiring and that adafruit-circuitpython-ina219 is "
              "installed) -- continuing with latency only")

    rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        try:
            row = benchmark_model(name, X, input_dim, num_classes,
                                  args.n_warmup, args.n_iters, args.n_runs,
                                  args.energy)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue
        for k, v in row.items():
            print(f"  {k:18s} {v}")
        rows.append(row)

    if rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
