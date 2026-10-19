import csv
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

def slug(field) -> str:
    """Regex helper function for make_run_name"""
    return re.sub(r"[^A-Za-z0-9\-]+", "-", str(field)).strip("-")

def make_run_name(
    train: str,
    model: str,
    pretrained: str,
    prompt_key: str,
    extras: str = ""
) -> str: 
    """Naming conventions: {train}_{model}_{pretrained}_{prompt}[_{Extras}]
    
    ex1) make_run_name("zs", "ViT-B-32", "openai", "p1") -> 'zs_ViT-B-32_openai_p1'
    ex2) make_run_name("ft-full", "ViT-B-32", "openai", "p1", "lr1e-5-ep10-s0") -> 'ft-full_ViT-B-32_openai_p1_lr1e-5-ep10-s0'
    """
    fields = [train, model, pretrained, prompt_key] + ([extras] if extras else [])


def git_commit():
    """Short hash of the code that produced the run, plus '+dirty' if there are uncommitted changes; None if no repo exists"""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        if sha.returncode != 0:
            return None
        changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return sha.stdout.strip() + ("+dirty" if changes else "")
    except FileNotFoundError:
        return None

def clean(value):
    """Prep value to be safely written as a JSON/CSV"""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"): # for numpy scalars
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def save_run(
    run_name: str,
    settings: dict,
    metrics: dict,
    out_dir="runs",
    overwrite: bool = False
) -> Path:
    folder = Path(out_dir) / run_name

    # Don't silently overwrite a run's metrics
    if folder.exists() and not overwrite:
        raise FileExistsError(
            f"{folder} already exists. Either pick a different run_name, delete this folder, or set overwrite=True"
        )
    folder.mkdir(parents=True, exist_ok=True)

    record = {
        "run_name": run_name,
        "date": datetime.now().isoformat(timespec="seonds"),
        "git_commit": git_commit(),
        "settings": settings,
        "metrics": metrics,
    }

    # Write records to a JSON
    with open(folder / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean(record), f, indent=2, default=str)

    # Write records to a csv
    rows = [{"type": name, **entry} for name, entry in metrics["per_type"].items()]
    with open(folder / "per_type.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(clean(r) for r in rows)

    return folder

def fmt(x) -> str:
    """print_summary helper function to handle edge values"""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
 
def print_summary(metrics: dict) -> None:
    """Print all metrics to the terminal"""
    per_type = metrics["per_type"]
    columns = [c for c in next(iter(per_type.values())) if c != "n_relevant"]
    name_width = max(len(name) for name in per_type)
 
    print(f"mAP                   {fmt(metrics['mAP'])}")
    print(f"image->text recall@1  {fmt(metrics['image_to_text_recall@1'])}   ({metrics['n_tiles']} tiles)")
    print()
    header = f"{'type':<{name_width}}  {'n':>4}  " + "  ".join(f"{c:>12}" for c in columns)
    print(header)
    print("-" * len(header))
    for name, entry in per_type.items():
        cells = "  ".join(f"{fmt(entry[c]):>12}" for c in columns)
        print(f"{name:<{name_width}}  {entry['n_relevant']:>4}  {cells}")
