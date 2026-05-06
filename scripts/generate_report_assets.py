from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ESTIMATORS = ["ips", "snips", "dm", "dr"]


POLICY_LABELS = {
    "uniform_random": "Uniform Random",
    "linucb_alpha_0.5": "LinUCB",
    "logistic_ucb_alpha_0.5": "Logistic UCB",
    "linear_thompson_sampling": "Linear TS",
}


FAMILY_LABELS = {
    "uniform_random": "Uniform Random",
    "epsilon_greedy_popularity": "Eps-Greedy",
    "linucb": "LinUCB",
    "logistic_ucb": "Logistic UCB",
    "linear_thompson_sampling": "Linear TS",
}


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    configure_plots()

    experiment = read_csv(project_root / args.experiment_results)
    sweep = read_csv(project_root / args.policy_sweep_results)
    metadata = read_metadata(project_root / args.training_metadata)
    training_log = read_optional_csv(project_root / args.training_log)

    experiment = add_policy_labels(experiment)
    sweep = add_sweep_labels(sweep)

    save_dataset_summary(metadata, tables_dir)
    save_core_tables(experiment, sweep, tables_dir)

    plot_policy_value_bars(experiment, figures_dir)
    plot_estimator_comparison(experiment, figures_dir)
    plot_policy_value_ci(experiment, figures_dir)
    plot_policy_diagnostics(experiment, figures_dir)
    plot_sweep_curves(sweep, figures_dir)
    plot_obp_parity(experiment, figures_dir)
    if training_log is not None and metadata.get("reward_model_backend") == "torch":
        plot_training_loss(training_log, figures_dir)

    manifest = build_manifest(output_dir)
    write_json(manifest, output_dir / "manifest.json")
    print(f"Saved report assets to {output_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Tables:  {tables_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready figures and tables.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--output-dir", default="report_assets", help="Directory for generated assets.")
    parser.add_argument("--experiment-results", default="experiment_results.csv")
    parser.add_argument("--policy-sweep-results", default="policy_sweep_results.csv")
    parser.add_argument("--training-metadata", default="training_artifacts/training_metadata.json")
    parser.add_argument("--training-log", default="training_artifacts/reward_model_training_log.csv")
    return parser.parse_args()


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_policy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["policy_label"] = df["policy"].apply(policy_label)
    return df


def add_sweep_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["policy_family_label"] = df["policy_family"].map(FAMILY_LABELS).fillna(df["policy_family"])
    df["policy_label"] = df["policy"].apply(policy_label)
    return df


def policy_label(policy: str) -> str:
    if policy in POLICY_LABELS:
        return POLICY_LABELS[policy]
    if policy.startswith("epsilon_greedy_popularity"):
        return "Eps-Greedy Popularity"
    if policy.startswith("linucb"):
        return "LinUCB"
    if policy.startswith("logistic_ucb"):
        return "Logistic UCB"
    if policy.startswith("linear_thompson_sampling"):
        return "Linear TS"
    return policy


def save_dataset_summary(metadata: dict, tables_dir: Path) -> None:
    summary = metadata.get("dataset_summary", {})
    rows = [
        ("Total logged rounds", summary.get("n_rounds")),
        ("Training rounds", summary.get("train_rounds")),
        ("Evaluation rounds", summary.get("eval_rounds")),
        ("Number of actions", summary.get("n_actions")),
        ("Context dimension", summary.get("context_dim")),
        ("Action context dimension", summary.get("action_context_dim")),
        ("Evaluation behavior mean reward", summary.get("behavior_mean_reward_eval")),
        ("Reward model backend", metadata.get("reward_model_backend")),
        ("Device used", metadata.get("device_used")),
        ("Bootstrap samples", metadata.get("core_bootstrap_samples")),
    ]
    df = pd.DataFrame(rows, columns=["item", "value"]).dropna()
    save_table(df, tables_dir / "dataset_summary")


def save_core_tables(experiment: pd.DataFrame, sweep: pd.DataFrame, tables_dir: Path) -> None:
    core_columns = [
        "policy_label",
        "behavior_mean_reward",
        "ips",
        "snips",
        "dm",
        "dr",
        "effective_sample_size_ratio",
    ]
    core = experiment[core_columns].copy()
    core = round_numeric(core, 6)
    save_table(core, tables_dir / "core_policy_values")

    ci_columns = ["policy_label"]
    for estimator in ESTIMATORS:
        ci_columns.extend([estimator, f"{estimator}_ci_lower", f"{estimator}_ci_upper"])
    available_ci_columns = [column for column in ci_columns if column in experiment.columns]
    ci_table = round_numeric(experiment[available_ci_columns].copy(), 6)
    save_table(ci_table, tables_dir / "core_policy_confidence_intervals")

    sweep_summary = (
        sweep.sort_values(["policy_family", "dr"], ascending=[True, False])
        .groupby("policy_family", as_index=False)
        .first()
    )
    sweep_columns = [
        "policy_family_label",
        "sweep_parameter",
        "sweep_value",
        "policy",
        "ips",
        "snips",
        "dm",
        "dr",
        "effective_sample_size_ratio",
    ]
    sweep_summary = round_numeric(sweep_summary[sweep_columns], 6)
    save_table(sweep_summary, tables_dir / "best_sweep_by_dr")

    diagnostics = experiment[
        [
            "policy_label",
            "mean_importance_weight",
            "effective_sample_size",
            "effective_sample_size_ratio",
        ]
    ].copy()
    diagnostics = round_numeric(diagnostics, 6)
    save_table(diagnostics, tables_dir / "ope_diagnostics")


def save_table(df: pd.DataFrame, base_path: Path) -> None:
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_latex(base_path.with_suffix(".tex"), index=False, escape=True)


def round_numeric(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_columns] = df[numeric_columns].round(digits)
    return df


def plot_policy_value_bars(experiment: pd.DataFrame, figures_dir: Path) -> None:
    long_df = experiment.melt(
        id_vars=["policy_label"],
        value_vars=ESTIMATORS,
        var_name="Estimator",
        value_name="Estimated Policy Value",
    )
    long_df["Estimator"] = long_df["Estimator"].str.upper()

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    sns.barplot(data=long_df, x="policy_label", y="Estimated Policy Value", hue="Estimator", ax=ax)
    ax.axhline(experiment["behavior_mean_reward"].iloc[0], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("Estimated policy value")
    ax.set_title("Policy Value Estimates by OPE Estimator")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Estimator", ncol=4, frameon=False)
    save_figure(fig, figures_dir / "core_policy_value_by_estimator")


def plot_estimator_comparison(experiment: pd.DataFrame, figures_dir: Path) -> None:
    heatmap_df = experiment.set_index("policy_label")[ESTIMATORS]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Estimated value"},
        ax=ax,
    )
    ax.set_xlabel("Estimator")
    ax.set_ylabel("")
    ax.set_title("OPE Estimate Matrix")
    save_figure(fig, figures_dir / "ope_estimator_heatmap")


def plot_policy_value_ci(experiment: pd.DataFrame, figures_dir: Path) -> None:
    estimators_with_ci = [name for name in ESTIMATORS if f"{name}_ci_lower" in experiment.columns]
    if not estimators_with_ci:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    axes = axes.flatten()
    for ax, estimator in zip(axes, estimators_with_ci):
        values = experiment[estimator].to_numpy(dtype=float)
        lower = experiment[f"{estimator}_ci_lower"].to_numpy(dtype=float)
        upper = experiment[f"{estimator}_ci_upper"].to_numpy(dtype=float)
        yerr = np.vstack([values - lower, upper - values])
        x = np.arange(len(experiment))
        ax.errorbar(x, values, yerr=yerr, fmt="o", capsize=4, linewidth=1.5)
        ax.axhline(experiment["behavior_mean_reward"].iloc[0], color="black", linestyle="--", linewidth=1.0)
        ax.set_title(estimator.upper())
        ax.set_ylabel("Policy value")
        ax.set_xticks(x)
        ax.set_xticklabels(experiment["policy_label"], rotation=35, ha="right")
    for ax in axes[len(estimators_with_ci) :]:
        ax.axis("off")
    fig.suptitle("Bootstrap Confidence Intervals", y=1.02, fontweight="bold")
    save_figure(fig, figures_dir / "core_policy_value_confidence_intervals")


def plot_policy_diagnostics(experiment: pd.DataFrame, figures_dir: Path) -> None:
    diagnostics = experiment.melt(
        id_vars=["policy_label"],
        value_vars=["effective_sample_size_ratio"],
        var_name="Diagnostic",
        value_name="Value",
    )
    diagnostics["Diagnostic"] = diagnostics["Diagnostic"].map(
        {
            "effective_sample_size_ratio": "ESS ratio",
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    sns.barplot(data=diagnostics, x="policy_label", y="Value", hue="Diagnostic", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, max(1.0, diagnostics["Value"].max() * 1.1))
    ax.set_title("Off-Policy Evaluation Stability Diagnostics")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="", frameon=False)
    save_figure(fig, figures_dir / "ope_stability_diagnostics")


def plot_sweep_curves(sweep: pd.DataFrame, figures_dir: Path) -> None:
    sweep = sweep[sweep["sweep_parameter"] != "none"].copy()
    if sweep.empty:
        return

    for estimator in ["ips", "snips", "dr"]:
        grid = sns.FacetGrid(
            sweep,
            col="policy_family_label",
            col_wrap=2,
            sharex=False,
            sharey=False,
            height=3.0,
            aspect=1.35,
        )
        grid.map_dataframe(
            sns.lineplot,
            x="sweep_value",
            y=estimator,
            marker="o",
            linewidth=1.8,
        )
        grid.set_axis_labels("Sweep value", f"{estimator.upper()} estimate")
        grid.set_titles("{col_name}")
        grid.fig.suptitle(f"Policy Sweep by {estimator.upper()} Estimate", y=1.03, fontweight="bold")
        save_figure(grid.fig, figures_dir / f"policy_sweep_{estimator}")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.lineplot(
        data=sweep,
        x="sweep_value",
        y="effective_sample_size_ratio",
        hue="policy_family_label",
        marker="o",
        linewidth=1.8,
        ax=ax,
    )
    ax.set_xlabel("Sweep value")
    ax.set_ylabel("ESS ratio")
    ax.set_title("Effective Sample Size Across Policy Sweeps")
    ax.legend(title="Policy family", frameon=False)
    save_figure(fig, figures_dir / "policy_sweep_effective_sample_size")


def plot_obp_parity(experiment: pd.DataFrame, figures_dir: Path) -> None:
    diff_columns = [column for column in experiment.columns if column.startswith("obp_") and column.endswith("_abs_diff")]
    if not diff_columns:
        return
    parity = experiment.melt(
        id_vars=["policy_label"],
        value_vars=diff_columns,
        var_name="Estimator",
        value_name="Absolute difference",
    )
    parity["Estimator"] = (
        parity["Estimator"]
        .str.replace("obp_", "", regex=False)
        .str.replace("_abs_diff", "", regex=False)
        .str.upper()
    )
    parity["Absolute difference"] = parity["Absolute difference"].clip(lower=1e-18)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    sns.barplot(data=parity, x="policy_label", y="Absolute difference", hue="Estimator", ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Manual vs. OBP absolute difference")
    ax.set_title("Parity Check Against OBP")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Estimator", ncol=4, frameon=False)
    save_figure(fig, figures_dir / "obp_parity_abs_diff")


def plot_training_loss(training_log: pd.DataFrame, figures_dir: Path) -> None:
    if not {"epoch", "loss"}.issubset(training_log.columns):
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    sns.lineplot(data=training_log, x="epoch", y="loss", linewidth=1.8, ax=ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title("Reward Model Training Loss")
    save_figure(fig, figures_dir / "reward_model_training_loss")


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_manifest(output_dir: Path) -> dict:
    files = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file())
    return {"output_dir": str(output_dir), "files": files}


if __name__ == "__main__":
    main()
