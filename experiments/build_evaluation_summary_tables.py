
"""Build Markdown and LaTeX manuscript tables from evaluation metrics JSON."""

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "final_evaluation_metrics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

# Core fields expected in the current evaluation results.
# Optional fields are included automatically when present.
METRICS = [
    ("Accuracy_PA", "Accuracy (Point-Adjusted)", "percent", True),
    ("Precision_PA", "Precision (Point-Adjusted)", "percent", True),
    ("Recall_PA", "Recall (Point-Adjusted)", "percent", True),
    ("F1_Score_PA", "F1-score (Point-Adjusted)", "percent", True),
    ("ROC_AUC", "ROC-AUC", "percent", True),
    ("PR_AUC", "PR-AUC", "percent", False),
    ("Event_Detection_Rate", "Event Detection Rate", "percent", False),
    ("Mean_Detection_Delay", "Mean Detection Delay (frames)", "decimal", False),
    ("Detected_Events", "Detected Events", "integer", False),
    ("Total_Events", "Total Events", "integer", False),
    ("False_Alarm_Events", "False Alarm Events", "integer", False),
    ("Threshold", "Decision Threshold", "threshold", True),
]


def load_metrics(input_path):
    """Read and validate the metrics JSON."""
    try:
        with input_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Metrics file not found: {input_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("The JSON root must be an object of metric names and values.")

    required = [key for key, _, _, is_required in METRICS if is_required]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(
            "Required metrics are missing from the JSON: " + ", ".join(missing)
        )

    rows = []
    for key, label, kind, is_required in METRICS:
        if key not in data:
            continue

        value = data[key]

        # These manuscript rows must contain scalar numeric values.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if is_required:
                raise ValueError(f"Metric '{key}' must be numeric.")
            continue

        if not math.isfinite(value):
            raise ValueError(f"Metric '{key}' must be finite.")

        if kind == "percent" and not 0 <= value <= 1:
            raise ValueError(
                f"Metric '{key}' must be between 0 and 1 before percentage formatting."
            )

        if kind == "integer" and not float(value).is_integer():
            raise ValueError(f"Metric '{key}' must be a whole number.")

        rows.append((label, format_value(value, kind)))

    return rows


def format_value(value, kind):
    """Format values for a manuscript table without changing source data."""
    if kind == "percent":
        return f"{value * 100:.2f}\\%"
    if kind == "threshold":
        return f"{value:.4f}"
    if kind == "integer":
        return str(int(value))
    return f"{value:.2f}"


def build_markdown(rows, source_name):
    lines = [
        "# Manuscript Table: Model Performance",
        "",
        f"Source: `{source_name}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    lines.append("")
    return "\n".join(lines)


def latex_escape(text):
    """Escape special LaTeX characters in text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_latex(rows, source_name):
    lines = [
        "% Automatically generated. Do not edit by hand.",
        f"% Source: {source_name}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model performance on the evaluation set}",
        r"\label{tab:model-performance}",
        r"\begin{tabular}{lr}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\hline",
    ]

    for label, value in rows:
        lines.append(f"{latex_escape(label)} & {value} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript tables from evaluation metrics JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the evaluation metrics JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated Markdown and LaTeX tables.",
    )
    args = parser.parse_args()

    try:
        rows = load_metrics(args.input)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = args.output_dir / "manuscript_tables.md"
        latex_path = args.output_dir / "manuscript_tables.tex"

        source_name = args.input.name
        markdown_path.write_text(
            build_markdown(rows, source_name), encoding="utf-8"
        )
        latex_path.write_text(
            build_latex(rows, source_name), encoding="utf-8"
        )

    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Generated Markdown table: {markdown_path}")
    print(f"Generated LaTeX table:    {latex_path}")
    print(f"Metrics included: {len(rows)}")


if __name__ == "__main__":
    main()

"""Generate manuscript tables in Markdown and LaTeX from evaluation metrics."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results" / "final_evaluation_metrics.json"
DEFAULT_OUTPUT = ROOT / "results"

METRICS = [
    ("Accuracy_PA", "Accuracy (Point-Adjusted)", "fraction", True),
    ("Precision_PA", "Precision (Point-Adjusted)", "fraction", True),
    ("Recall_PA", "Recall (Point-Adjusted)", "fraction", True),
    ("F1_Score_PA", "F1-score (Point-Adjusted)", "fraction", True),
    ("ROC_AUC", "ROC-AUC", "decimal", True),
    ("PR_AUC", "PR-AUC", "decimal", False),
    ("Event_Detection_Rate", "Event Detection Rate", "fraction", False),
    ("Mean_Detection_Delay", "Mean Detection Delay (frames)", "decimal", False),
    ("Detected_Events", "Detected Events", "integer", False),
    ("Total_Events", "Total Events", "integer", False),
    ("False_Alarm_Events", "False Alarm Events", "integer", False),
    ("Threshold", "Decision Threshold", "decimal", True),
]


def load_metrics(path):
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read metrics from {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("The JSON root must be an object.")

    required = [key for key, _, _, req in METRICS if req]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Missing required metrics: " + ", ".join(missing))

    rows = []
    for key, label, kind, required_metric in METRICS:
        if key not in data:
            continue

        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if required_metric:
                raise ValueError(f"Metric {key} must be numeric.")
            continue

        if not math.isfinite(value):
            raise ValueError(f"Metric {key} must be finite.")

        if kind in ("fraction", "decimal") and kind == "fraction":
            if not 0 <= value <= 1:
                raise ValueError(f"Metric {key} must be between 0 and 1.")

        if kind == "integer" and not float(value).is_integer():
            raise ValueError(f"Metric {key} must be a whole number.")

        rows.append((label, value, kind))

    return rows


def format_markdown(value, kind):
    if kind == "fraction":
        return f"{value * 100:.2f}%"
    if kind == "integer":
        return str(int(value))
    return f"{value:.4f}"


def format_latex(value, kind):
    if kind == "fraction":
        return f"{value * 100:.2f}\\%"
    if kind == "integer":
        return str(int(value))
    return f"{value:.4f}"


def build_markdown(rows, source):
    lines = [
        "# Manuscript Table: Model Performance",
        "",
        f"Source: `{source}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for label, value, kind in rows:
        lines.append(f"| {label} | {format_markdown(value, kind)} |")
    lines.append("")
    return "\n".join(lines)


def escape_latex(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_latex(rows, source):
    lines = [
        "% Automatically generated. Do not edit by hand.",
        f"% Source: {source}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model performance on the evaluation set}",
        r"\label{tab:model-performance}",
        r"\begin{tabular}{lr}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\hline",
    ]
    for label, value, kind in rows:
        lines.append(
            f"{escape_latex(label)} & {format_latex(value, kind)} " + r"\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript tables from metrics JSON."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        rows = load_metrics(args.input)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        md_path = args.output_dir / "manuscript_tables.md"
        tex_path = args.output_dir / "manuscript_tables.tex"

        md_path.write_text(
            build_markdown(rows, args.input.name), encoding="utf-8"
        )
        tex_path.write_text(
            build_latex(rows, args.input.name), encoding="utf-8"
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Generated Markdown table: {md_path}")
    print(f"Generated LaTeX table: {tex_path}")
    print(f"Metrics included: {len(rows)}")


if __name__ == "__main__":
    main()
