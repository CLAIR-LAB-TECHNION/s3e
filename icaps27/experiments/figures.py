# SPDX-License-Identifier: MIT
"""Regenerate the paper's tables and figures from ``results/raw`` (no model, no GPU).

Every artifact is produced only when its inputs exist, so this runs after
any subset of the experiments. Tables are written as Markdown and LaTeX
(booktabs); figures as PDF and PNG, each next to a CSV of its plotted data
(the table view of the figure). ``results/figures/MANIFEST.md`` lists what
was produced from which inputs.

    python experiments/figures.py --results results/raw --out results/figures
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from s3e_exp import metrics  # noqa: E402
from s3e_exp.store import JsonlStore, read_json  # noqa: E402

# Categorical order validated for light mode (blue, orange, aqua, yellow,
# magenta, green, violet, red); markers are the secondary encoding.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
TEXT, MUTED, GRID, REFERENCE = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def _rel(path) -> str:
    """A source path relative to the artifact root (no absolute paths in outputs)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


class Outputs:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.manifest: list[str] = []

    def table(self, name: str, header: list[str], rows: list[list], caption: str,
              sources: list) -> None:
        def fmt(value):
            if isinstance(value, float):
                return "--" if math.isnan(value) else f"{value:.3f}"
            return str(value)

        cells = [[fmt(v) for v in row] for row in rows]
        md = [f"<!-- {caption} -->", "| " + " | ".join(header) + " |",
              "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in cells]
        (self.root / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        esc = [[c.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&") for c in r]
               for r in [header] + cells]
        tex = [r"\begin{table}[t]", r"\centering\small",
               r"\begin{tabular}{" + "l" * 1 + "r" * (len(header) - 1) + "}", r"\toprule",
               " & ".join(esc[0]) + r" \\", r"\midrule"] + \
            [" & ".join(r) + r" \\" for r in esc[1:]] + \
            [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}"]
        (self.root / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        self.manifest.append(f"- `{name}.md/.tex`: {caption} (from {', '.join(map(_rel, sources))})")

    def figure(self, name: str, fig, data: list[dict], caption: str, sources: list) -> None:
        fig.savefig(self.root / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.root / f"{name}.png", dpi=200, bbox_inches="tight")
        if data:
            fields = list(dict.fromkeys(k for row in data for k in row))
            with (self.root / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
        self.manifest.append(f"- `{name}.pdf/.png/.csv`: {caption} (from {', '.join(map(_rel, sources))})")

    def write_manifest(self) -> None:
        (self.root / "MANIFEST.md").write_text(
            "# Generated tables and figures\n\n" + "\n".join(self.manifest) + "\n", encoding="utf-8")


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return plt


def mean_std(values):
    s = metrics.summarize(values)
    return s["mean"], s["std"]


def rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else math.nan


# --- E1 ---


def e1_offline(results: Path, out: Outputs, plt) -> None:
    files = sorted((results / "e1").glob("plan-*.jsonl"))
    everything = [r for f in files for r in JsonlStore(f).read()]
    for domain in sorted({r.get("domain", "unknown") for r in everything}):
        rows = [r for r in everything if r.get("domain", "unknown") == domain]
        _e1_domain(domain, rows, files, out, plt)


def _e1_domain(domain: str, rows: list[dict], files, out: Outputs, plt) -> None:
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["model"], r["decoder"])][r.get("seed")].append(r)
    table = []
    for (model, decoder), by_seed in sorted(groups.items()):
        cells = [model, decoder, sum(len(v) for v in by_seed.values()), len(by_seed)]
        for key in ("valid", "executable", "exact"):
            m, s = mean_std([rate(v, key) for v in by_seed.values()])
            cells.append(f"{m:.3f} ± {s:.3f}")
        acc = [1 - sum(r["atom_errors"] for r in v) / sum(r["atoms"] for r in v)
               for v in by_seed.values()]
        m, s = mean_std(acc)
        cells.append(f"{m:.3f} ± {s:.3f}")
        table.append(cells)
    out.table(f"e1_planning-{domain}", ["model", "decoder", "scenes", "seeds", "valid plan",
                                        "executable", "exact state", "atom accuracy"], table,
              f"E1 ({domain}): open-loop planning from estimated states "
              "(mean ± sd across seeds; sd 0 when there is one seed).", files)

    # Figure: valid-plan rate vs number of blocks (threshold decoder) per model,
    # and atom accuracy vs valid rate with the oracle-noise reference.
    models = sorted({r["model"] for r in rows if r["model"] != "oracle"})
    sizes = sorted({r["size"] for r in rows if r.get("size") is not None})
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    data = []
    for i, model in enumerate(models[:len(SERIES)]):
        ys = []
        for size in sizes:
            subset = [r for r in rows if r["model"] == model and r["decoder"] == "threshold"
                      and r["size"] == size]
            ys.append(rate(subset, "valid"))
            data.append({"panel": "a", "model": model, "size": size, "valid": ys[-1]})
        left.plot(sizes, ys, color=SERIES[i], marker=MARKERS[i], label=model)
    oracle = [rate([r for r in rows if r["decoder"] == "oracle" and r["size"] == s], "valid")
              for s in sizes]
    left.plot(sizes, oracle, color=REFERENCE, linestyle="-", linewidth=1, label="true state")
    left.set_xlabel("blocks" if domain == "blocksworld" else "objects")
    left.set_ylabel("valid plans")
    left.set_ylim(-0.02, 1.02)
    left.set_xticks(sizes)
    left.legend(loc="lower left")
    noise = sorted({r["decoder"] for r in rows if r["decoder"].startswith("noise-")},
                   key=lambda d: float(d.split("-", 1)[1]))
    nx, ny = [], []
    for decoder in noise:
        subset = [r for r in rows if r["decoder"] == decoder]
        nx.append(1 - sum(r["atom_errors"] for r in subset) / sum(r["atoms"] for r in subset))
        ny.append(rate(subset, "valid"))
        data.append({"panel": "b", "model": "oracle", "decoder": decoder, "accuracy": nx[-1],
                     "valid": ny[-1]})
    if nx:
        right.plot([1.0] + nx, [oracle_rate(rows)] + ny, color=REFERENCE, linewidth=1,
                   label="true state, flipped atoms")
    for i, model in enumerate(models[:len(SERIES)]):
        for j, decoder in enumerate(("threshold", "calibrated", "map", "calibrated+map")):
            subset = [r for r in rows if r["model"] == model and r["decoder"] == decoder]
            if not subset:
                continue
            x = 1 - sum(r["atom_errors"] for r in subset) / sum(r["atoms"] for r in subset)
            y = rate(subset, "valid")
            right.scatter([x], [y], color=SERIES[i], marker=MARKERS[j], s=28, zorder=3,
                          edgecolors="#fcfcfb", linewidths=1.0,
                          label=decoder if i == 0 else None)
            data.append({"panel": "b", "model": model, "decoder": decoder, "accuracy": x,
                         "valid": y})
    right.set_xlabel("atom accuracy")
    right.set_ylabel("valid plans")
    right.set_ylim(-0.02, 1.02)
    decoders = right.legend(loc="upper left", title="marker = decoder", title_fontsize=7)
    if len(models) > 1:
        from matplotlib.lines import Line2D

        right.add_artist(decoders)
        right.legend(handles=[Line2D([], [], color=SERIES[i], marker="o", linestyle="",
                                     label=m) for i, m in enumerate(models[:len(SERIES)])],
                     loc="lower right", title="color = model", title_fontsize=7)
    fig.tight_layout()
    out.figure(f"e1_planning-{domain}", fig, data,
               f"E1 ({domain}): valid plans vs objects; valid plans vs atom accuracy", files)
    plt.close(fig)


def oracle_rate(rows):
    return rate([r for r in rows if r["decoder"] == "oracle"], "valid")


def e1_online(results: Path, out: Outputs, plt) -> None:
    files = sorted((results / "e1" / "online").rglob("episodes.jsonl"))
    rows = [r for f in files for r in JsonlStore(f).read()]
    if not rows:
        return
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("domain", "?"), r["model"], r["policy"], r["check"], r["decoder"],
                r.get("mask", "none"), "yes" if r.get("calibrated") else "no")].append(r)
    table = []
    for key, g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        table.append([*key, len(g), rate(g, "success"),
                      sum(r["steps"] for r in g) / len(g), sum(r["replans"] for r in g) / len(g),
                      sum(r.get("unplannable_estimates", 0) for r in g) / len(g),
                      sum(r["queries"] for r in g) / len(g),
                      sum(r["vlm_seconds"] for r in g) / len(g)])
    out.table("e1_online", ["domain", "model", "policy", "check", "decoder", "mask", "calibrated",
                            "episodes", "success", "steps", "replans", "unplannable estimates",
                            "VLM queries", "VLM seconds"], table,
              "E1: closed-loop execution with re-estimation (per episode means).", files)


# --- E2 ---


def e2(results: Path, out: Outputs, plt) -> None:
    count_files = sorted((results / "e2").glob("count-*.jsonl"))
    counts = [r for f in count_files for r in JsonlStore(f).read()]
    if counts:
        by = defaultdict(list)
        for r in counts:
            by[(r["domain"], r["objects"])].append(r)
        table = [[d, n, len(g)] + [sum(r[f"atoms_{m}"] for r in g) / len(g)
                                   for m in ("none", "dynamic", "relevant")]
                 + [sum(r["ground_and_translate_seconds"] for r in g) / len(g)]
                 for (d, n), g in sorted(by.items())]
        out.table("e2_atoms", ["domain", "objects", "scenes", "atoms (all)", "atoms (dynamic)",
                               "atoms (relevant)", "ground+translate s"], table,
                  "E2: grounded atoms per scene under each mask.", count_files)
        domains = sorted({r["domain"] for r in counts})
        fig, axes = plt.subplots(1, len(domains), figsize=(2.6 * len(domains), 2.3), squeeze=False)
        data = []
        for ax, domain in zip(axes[0], domains):
            xs = sorted({n for d, n in by if d == domain})
            for i, mask in enumerate(("none", "dynamic", "relevant")):
                ys = [sum(r[f"atoms_{mask}"] for r in by[(domain, n)]) / len(by[(domain, n)])
                      for n in xs]
                ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], label=mask)
                data += [{"domain": domain, "objects": x, "mask": mask, "atoms": y}
                         for x, y in zip(xs, ys)]
            ax.set_title(domain, color=TEXT)
            ax.set_xlabel("objects")
            ax.set_ylabel("atoms queried")
            if max(max(r["atoms_none"] for r in counts if r["domain"] == domain), 1) > 500:
                ax.set_yscale("log")
        axes[0][0].legend(title="mask", title_fontsize=7)
        fig.tight_layout()
        out.figure("e2_atoms", fig, data, "E2: atoms per scene vs objects, by mask", count_files)
        plt.close(fig)

    time_files = sorted((results / "e2").glob("time-*.jsonl"))
    timing = [r for f in time_files for r in JsonlStore(f).read()]
    # Scenes that ran out of memory at the requested batch size were retried
    # with smaller batches; they would mix batch sizes, so they are left out.
    timing = [r for r in timing if r.get("effective_batch_size", r["batch_size"]) == r["batch_size"]]
    if not timing:
        return
    groups = defaultdict(list)
    for r in timing:
        groups[(r["model"], r["backend"], r["batch_size"], r["mask"], r["size"])].append(r)
    table = []
    for key, g in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        m, s = mean_std([r["seconds"] for r in g])
        table.append([*key, sum(r["atoms"] for r in g) / len(g), f"{m:.3f} ± {s:.3f}",
                      sum(r["queries"] for r in g) / sum(r["seconds"] for r in g),
                      sum(r["inference_seconds"] for r in g) / sum(r["seconds"] for r in g),
                      max((r["peak_memory"]["bytes"] or 0) for r in g) / 1e9])
    out.table("e2_time", ["model", "backend", "batch", "mask", "blocks", "atoms", "s / scene",
                          "queries / s", "in-model share", "peak GB"], table,
              "E2: wall clock per scene (mean ± sd over scenes).", time_files)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    data = []
    series = sorted({(r["backend"], r["mask"]) for r in timing})
    for i, (backend, mask) in enumerate(series[:len(SERIES)]):
        batches = sorted({r["batch_size"] for r in timing if r["backend"] == backend
                          and r["mask"] == mask})
        sizes = sorted({r["size"] for r in timing})
        ys = []
        for size in sizes:
            # Each size at its fastest batch size (large batches can run out of memory).
            means = [sum(r["seconds"] for r in g) / len(g) for (m, b, bs, mk, sz), g in groups.items()
                     if b == backend and mk == mask and sz == size and g]
            ys.append(min(means) if means else math.nan)
        left.plot(sizes, ys, color=SERIES[i], marker=MARKERS[i], label=f"{backend}, {mask}")
        # Throughput over the sizes every batch size completed, so batch sizes
        # are compared on the same scenes.
        series_rows = [r for r in timing if r["backend"] == backend and r["mask"] == mask]
        shared = set.intersection(*[{r["size"] for r in series_rows if r["batch_size"] == b}
                                    for b in batches])
        qps = []
        for batch in batches:
            g = [r for r in series_rows if r["batch_size"] == batch and r["size"] in shared]
            qps.append(sum(r["queries"] for r in g) / sum(r["seconds"] for r in g))
        right.plot(batches, qps, color=SERIES[i], marker=MARKERS[i], label=f"{backend}, {mask}")
        data += [{"panel": "a", "backend": backend, "mask": mask, "size": s, "seconds": y}
                 for s, y in zip(sizes, ys)]
        data += [{"panel": "b", "backend": backend, "mask": mask, "batch": b, "qps": q}
                 for b, q in zip(batches, qps)]
    left.set_xlabel("blocks")
    left.set_ylabel("seconds per scene (fastest batch)")
    left.set_xticks(sorted({r["size"] for r in timing}))
    left.set_yscale("log")
    right.set_xlabel("batch size")
    right.set_ylabel("queries per second")
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    left.legend()
    fig.tight_layout()
    out.figure("e2_time", fig, data, "E2: seconds per scene vs blocks; throughput vs batch size",
               time_files)
    plt.close(fig)


# --- E3, E4, E5, E6, E7 ---


def e3(results: Path, out: Outputs, plt) -> None:
    files = sorted((results / "e3").glob("*.json"))
    table = []
    for f in files:
        report = read_json(f)
        for method, result in report["methods"].items():
            m, seeds = result["pooled"], result["across_seeds"]
            def cell(k, seeds=seeds, m=m):
                s = seeds.get(k, {})
                return f"{s['mean']:.3f} ± {s['std']:.3f}" if s.get("n", 0) > 1 else m.get(k)
            table.append([f.stem, method, cell("accuracy"), cell("f1"), cell("brier"),
                          cell("ece"), cell("no_answer_rate"), result["seconds_per_query"]])
    if table:
        out.table("e3_scoring", ["model", "method", "accuracy", "F1", "Brier", "ECE",
                                 "no answer", "s / query"], table,
                  "E3: token probabilities vs text matching vs a regex parser.", files)


def e4(results: Path, out: Outputs, plt) -> None:
    files = sorted((results / "e4").glob("scopes-*.json"))
    table, data = [], []
    for f in files:
        report = read_json(f)
        u = report["uncalibrated"]
        table.append([f.stem.removeprefix("scopes-"), "uncalibrated", u["accuracy"], u["f1"],
                      u["brier"], u["ece"], u["nll"]])
        for scope, result in report["scopes"].items():
            s = result["summary"]
            table.append([f.stem.removeprefix("scopes-"), scope] +
                         [s[k]["mean"] for k in ("accuracy", "f1", "brier", "ece", "nll")])
    if table:
        out.table("e4_scopes", ["model", "calibration", "accuracy", "F1", "Brier", "ECE", "NLL"],
                  table, "E4: Platt scaling by scope, cross-fitted over scenes.", files)
        fig, axes = plt.subplots(1, len(files), figsize=(2.4 * len(files), 2.3), squeeze=False)
        for ax, f in zip(axes[0], files):
            report = read_json(f)
            ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1)
            for i, (name, bins) in enumerate(report["reliability"].items()):
                xs = [b["confidence"] for b in bins if b["count"]]
                ys = [b["frequency"] for b in bins if b["count"]]
                ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], label=name)
                data += [{"model": f.stem, "calibration": name, "confidence": x, "frequency": y}
                         for x, y in zip(xs, ys)]
            ax.set_title(f.stem.removeprefix("scopes-"), color=TEXT)
            ax.set_xlabel("P(true)")
            ax.set_ylabel("frequency true")
        axes[0][0].legend()
        fig.tight_layout()
        out.figure("e4_reliability", fig, data, "E4: reliability before and after lifted Platt",
                   files)
        plt.close(fig)
    view_files = sorted((results / "e4").glob("views-*.json"))
    table = []
    for f in view_files:
        report = read_json(f)
        for count, result in report["by_count"].items():
            for mode, s in result["summary"].items():
                table.append([f.stem.removeprefix("views-"), f"{count} view(s)", mode] +
                             [s[k]["mean"] for k in ("accuracy", "f1", "ece", "exact_match")])
        if "all_in_one_prompt" in report:
            m = report["all_in_one_prompt"]
            table.append([f.stem.removeprefix("views-"), "all in one prompt", "raw",
                          m["accuracy"], m["f1"], m["ece"], m["exact_match"]])
    if table:
        out.table("e4_views", ["model", "views", "calibration", "accuracy", "F1", "ECE",
                               "exact state"], table,
                  "E4: averaging single-view estimates (mean over view subsets).", view_files)
    transfer_files = sorted((results / "e4").glob("transfer-*.json"))
    table = []
    for f in transfer_files:
        for split, result in read_json(f)["splits"].items():
            for condition in ("uncalibrated", "global", "lifted", "grounded"):
                m = result[condition]
                table.append([f.stem.removeprefix("transfer-"), split, condition, m["accuracy"],
                              m["brier"], m["ece"]])
    if table:
        out.table("e4_transfer", ["model", "split", "calibration", "accuracy", "Brier", "ECE"],
                  table, "E4: calibration transfer across problem instances.", transfer_files)


def e5(results: Path, out: Outputs, plt) -> None:
    files = sorted((results / "e5").glob("*.json"))
    table = []
    for f in files:
        for condition, r in read_json(f)["conditions"].items():
            m = r["pooled"]
            table.append([f.stem, condition, m["accuracy"], m["f1"], m["ece"],
                          r["answer_mass"]["mean"], r["reasoning_tokens"]["mean"],
                          r["seconds_per_query"]])
    if table:
        out.table("e5_reasoning", ["model", "condition", "accuracy", "F1", "ECE", "answer mass",
                                   "thinking tokens", "s / query"], table,
                  "E5: reasoning models in logprobs mode.", files)


def e6(results: Path, out: Outputs, plt) -> None:
    files = sorted((results / "e6").glob("compare-*.json"))
    table = []
    for f in files:
        report = read_json(f)
        for tool, m in report["tools"].items():
            agree = m.get("agreement_with_s3e", {})
            table.append([f.stem.removeprefix("compare-"), tool, m["accuracy"], m["f1"], m["brier"],
                          m["ece"], m.get("mean_answer_mass", math.nan), m["seconds_per_query"],
                          agree.get("same_decision", 1.0)])
    if table:
        out.table("e6_tools", ["model", "tool", "accuracy", "F1", "Brier", "ECE", "answer mass",
                               "s / query", "same decision as S3E"], table,
                  "E6: S3E and other scorers on identical prompts and images.", files)
    matrix = read_json(ROOT / "configs/feature_matrix.json")
    symbols = {True: "yes", False: "no", "partial": "partial", None: "?"}
    rows = [[tool["name"]] + [symbols.get(tool["features"].get(f["key"]), "?")
                              for f in matrix["features"]] for tool in matrix["tools"]]
    out.table("feature_matrix", ["tool"] + [f["short"] for f in matrix["features"]], rows,
              "Related tools: " + "; ".join(f"{f['short']}: {f['label']}"
                                            for f in matrix["features"]),
              ["configs/feature_matrix.json"])


def e7(results: Path, out: Outputs, plt) -> None:
    files = sorted((results / "e7").glob("*.summary.json"))
    for f in files:
        report = read_json(f)
        versions = report["versions"]
        names = list(versions)
        features = sorted({k for v in versions.values() for k in v["features"]})
        rows = [[key] + [versions[n].get(key) for n in names]
                for key in ("version", "public_names", "source_lines", "seconds_per_query",
                            "queries_per_second", "accuracy", "brier")
                if any(versions[n].get(key) is not None for n in names)]
        rows += [[feature] + ["yes" if versions[n]["features"].get(feature) else "no"
                              for n in names] for feature in features]
        out.table(f"e7_{f.stem.removesuffix('.summary')}", ["", *names], rows,
                  "E7: S3E versions on the same scenes, model, and prompts.", [f])
    workshop = sorted((results / "e7").glob("workshop-*.json"))
    table = []
    for f in workshop:
        report = read_json(f)
        ours = report["ours"]
        table.append([f.stem, "this run"] + [ours[c] for c in ("acc@0.3", "acc@0.5", "acc@0.7",
                                                                "AP_micro", "AP_macro")])
        if report.get("paper"):
            table.append([f.stem, f"2025 paper ({report['paper_row']})", *report["paper"]])
    if table:
        out.table("e7_workshop", ["setting", "source", "acc@0.3", "acc@0.5", "acc@0.7",
                                  "AP micro", "AP macro"], table,
                  "E7: the 2025 workshop setting with the current library.", workshop)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results/raw")
    parser.add_argument("--out", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args(argv)
    plt = style()
    out = Outputs(args.out)
    for step in (e1_offline, e1_online, e2, e3, e4, e5, e6, e7):
        step(args.results, out, plt)
    out.write_manifest()
    print(json.dumps(out.manifest, indent=1))


if __name__ == "__main__":
    main()
