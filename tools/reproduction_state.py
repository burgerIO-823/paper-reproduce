#!/usr/bin/env python3
"""
Reproduction State Manager

Defines the schema for structured paper parsing output and manages the
reproduction_state.yaml file that persists context across conversation turns.

Usage:
    # Initialize a new state file
    python reproduction_state.py init <task_id> --paper-title "..." [--arxiv-id ...]

    # Read current state
    python reproduction_state.py read <state_file>

    # Update a specific field
    python reproduction_state.py update <state_file> --set status=stage-2

    # Validate completeness
    python reproduction_state.py validate <state_file>
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    # Fallback: write JSON (valid YAML subset) if PyYAML not available
    yaml = None


# ── Schema definition ──────────────────────────────────────────────────────

PAPER_METADATA_SCHEMA = {
    "title": "",
    "authors": [],
    "arxiv_id": "",
    "venue": "",
    "year": "",
    "abstract": "",
    "categories": [],
    "source_url": "",
}

ARCHITECTURE_SCHEMA = {
    "backbone": "",            # e.g., "ResNet-50", "ViT-B/16", "BERT-base"
    "neck": "",                # e.g., "FPN", "PAFPN", "" if not applicable
    "head": "",                # e.g., "3-layer MLP", "Linear classifier"
    "loss_functions": [],      # e.g., ["CrossEntropy", "FocalLoss(gamma=2)"]
    "novel_components": [],    # New modules introduced by this paper
    "input_shape": "",         # e.g., "(3, 224, 224)", "(512,)"
    "output_shape": "",        # e.g., "(1000,)", "(B, L, vocab_size)"
    "architecture_diagram_refs": [],  # References to figures showing architecture
}

TRAINING_RECIPE_SCHEMA = {
    "optimizer": "",           # e.g., "SGD", "AdamW"
    "learning_rate": "",       # e.g., "0.1", "1e-3"
    "lr_schedule": "",         # e.g., "cosine annealing", "step decay at 30,60,80 epochs"
    "batch_size": "",          # e.g., "256", "32 per GPU × 8 GPUs"
    "epochs": "",              # e.g., "100", "300"
    "warmup_epochs": "",       # e.g., "5", "" if not mentioned
    "weight_decay": "",
    "momentum": "",
    "gradient_clip": "",
    "mixed_precision": "",     # e.g., "fp16", "bf16", "" if not mentioned
    "augmentations": [],       # e.g., ["RandomResizedCrop", "RandAugment"]
    "regularization": [],      # e.g., ["Dropout(0.1)", "LabelSmoothing(0.1)"]
    "ema": "",                 # Exponential moving average, e.g., "0.999"
    "additional_details": "",  # Free text for any other training details
}

DATASET_SCHEMA = {
    "name": "",
    "download_url": "",
    "preprocessing": [],       # e.g., ["Resize to 256", "CenterCrop 224"]
    "train_split": "",
    "val_split": "",
    "test_split": "",
    "num_classes": "",
    "notes": "",               # Any special notes about the dataset
}

EVALUATION_SCHEMA = {
    "metrics": [],             # e.g., ["Top-1 Accuracy", "Top-5 Accuracy"]
    "benchmarks": [],          # e.g., ["ImageNet", "CIFAR-10"]
    "comparison_methods": [],  # Methods compared against in the paper
    "reported_results": {},    # e.g., {"Top-1 Accuracy": "76.2%", "Top-5": "92.9%"}
}

HYPERPARAMS_SCHEMA = {
    "seed": "",
    "num_workers": "",
    "precision": "",           # e.g., "fp32", "fp16", "bf16"
    # Additional hyperparameters can be added dynamically
}

REPRODUCTION_STATE_TEMPLATE = {
    # ── Metadata ──
    "task_id": "",
    "paper_metadata": dict(PAPER_METADATA_SCHEMA),
    "status": "stage-1",
    "created_at": "",
    "updated_at": "",

    # ── Stage 1 outputs ──
    "parsed_architecture": dict(ARCHITECTURE_SCHEMA),
    "parsed_training_recipe": dict(TRAINING_RECIPE_SCHEMA),
    "parsed_dataset": dict(DATASET_SCHEMA),
    "parsed_evaluation": dict(EVALUATION_SCHEMA),
    "parsed_hyperparams": dict(HYPERPARAMS_SCHEMA),
    "parsed_unclear_areas": [],  # [{item, importance, inference_if_any}]

    # ── Stage 2 outputs ──
    "reproduction_plan": {
        "dependencies": [],
        "pretrained_weights": [],
        "missing_info": [],
        "implementation_steps": [],
        "known_pitfalls": [],
        "estimated_resources": "",
    },

    # ── Stage 3 outputs ──
    "generated_files": [],

    # ── Stage 4 outputs ──
    "experiment_results": {},

    # ── Stage 5 outputs ──
    "verification_report": {},

    # ── Decision log ──
    "decisions": [],  # [{timestamp, stage, decision, reason}]

    # ── Context summary (for fast recovery after context window overflow) ──
    "context_summary": {
        "what_was_done": "",
        "what_is_next": "",
        "key_findings": [],
        "pending_confirmations": [],
    },
}


# ── State file operations ──────────────────────────────────────────────────

def init_state(task_id: str, pdf_path: str = "", arxiv_id: str = "",
               paper_title: str = "", output_dir: str = ".") -> dict:
    """
    Initialize a new reproduction state file.

    Returns the state dict and writes reproduction_state.yaml to output_dir.
    """
    state = json.loads(json.dumps(REPRODUCTION_STATE_TEMPLATE))  # deep copy
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    state["task_id"] = task_id
    state["paper_metadata"]["arxiv_id"] = arxiv_id
    state["paper_metadata"]["title"] = paper_title
    state["created_at"] = now
    state["updated_at"] = now
    state["context_summary"]["what_is_next"] = "Complete Stage 1: Paper parsing and structured extraction"

    if pdf_path:
        state["pdf_path"] = pdf_path

    filepath = os.path.join(output_dir, "reproduction_state.yaml")
    _write_state(state, filepath)

    return state


def read_state(filepath: str) -> dict:
    """Read a reproduction state file. Returns the parsed dict."""
    if not os.path.exists(filepath):
        return {"success": False, "error": f"State file not found: {filepath}"}

    with open(filepath, "r", encoding="utf-8") as f:
        if yaml:
            state = yaml.safe_load(f)
        else:
            state = json.load(f)

    state["_filepath"] = filepath
    return state


def update_state(filepath: str, updates: dict) -> dict:
    """
    Apply updates to an existing state file.

    Updates are merged shallowly. For nested dicts, use dot-notation keys:
        update_state(fp, {"parsed_architecture.backbone": "ResNet-50"})
    """
    state = read_state(filepath)
    if not state.get("success", True):
        return state

    # Apply dot-notation updates
    for key, value in updates.items():
        if "." in key:
            parts = key.split(".")
            target = state
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            state[key] = value

    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_state(state, filepath)
    return state


def add_decision(filepath: str, stage: str, decision: str, reason: str = "") -> dict:
    """Append a decision entry to the state file's decision log."""
    state = read_state(filepath)
    if not state.get("success", True):
        return state

    state.setdefault("decisions", []).append({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "decision": decision,
        "reason": reason,
    })
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_state(state, filepath)
    return state


def update_context_summary(filepath: str, **kwargs) -> dict:
    """Update the context_summary section of the state file."""
    state = read_state(filepath)
    if not state.get("success", True):
        return state

    summary = state.setdefault("context_summary", {})
    summary.update(kwargs)
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_state(state, filepath)
    return state


def validate_state(filepath: str) -> dict:
    """
    Validate a reproduction state file for completeness.

    Returns a dict with:
        - success: bool
        - complete: bool
        - missing: list of required fields that are empty
        - warnings: list of non-critical issues
    """
    state = read_state(filepath)
    if not state.get("success", True):
        return state

    missing = []
    warnings = []

    # Check paper metadata
    meta = state.get("paper_metadata", {})
    for field in ["title", "authors"]:
        if not meta.get(field):
            missing.append(f"paper_metadata.{field}")

    # Check parsed sections based on current stage
    status = state.get("status", "")
    if status in ("stage-2", "stage-3", "stage-4", "stage-5", "completed"):
        arch = state.get("parsed_architecture", {})
        if not arch.get("backbone"):
            warnings.append("parsed_architecture.backbone is empty")

        train = state.get("parsed_training_recipe", {})
        for field in ["optimizer", "learning_rate", "batch_size"]:
            if not train.get(field):
                warnings.append(f"parsed_training_recipe.{field} is empty")

    if status in ("stage-3", "stage-4", "stage-5", "completed"):
        plan = state.get("reproduction_plan", {})
        if not plan.get("implementation_steps"):
            missing.append("reproduction_plan.implementation_steps")

    return {
        "success": True,
        "complete": len(missing) == 0,
        "missing": missing,
        "warnings": warnings,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _write_state(state: dict, filepath: str):
    """Write state dict to file, removing internal keys."""
    write_state = {k: v for k, v in state.items() if not k.startswith("_")}

    with open(filepath, "w", encoding="utf-8") as f:
        if yaml:
            yaml.dump(write_state, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            json.dump(write_state, f, ensure_ascii=False, indent=2)


def _parse_updates(raw_updates: list) -> dict:
    """Parse --set key=value pairs into a dict."""
    updates = {}
    for item in raw_updates:
        if "=" in item:
            key, value = item.split("=", 1)
            # Try to parse as JSON for typed values
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
            updates[key] = value
    return updates


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manage reproduction state files"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize a new state file")
    init_parser.add_argument("task_id", help="Unique task identifier")
    init_parser.add_argument("--paper-title", default="", help="Paper title")
    init_parser.add_argument("--arxiv-id", default="", help="arXiv ID")
    init_parser.add_argument("--pdf-path", default="", help="Path to the paper PDF")
    init_parser.add_argument("--output-dir", "-o", default=".", help="Output directory")

    # read
    read_parser = subparsers.add_parser("read", help="Read a state file")
    read_parser.add_argument("filepath", help="Path to reproduction_state.yaml")

    # update
    update_parser = subparsers.add_parser("update", help="Update fields in a state file")
    update_parser.add_argument("filepath", help="Path to reproduction_state.yaml")
    update_parser.add_argument("--set", action="append", default=[], help="key=value pairs (repeatable)")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate state completeness")
    validate_parser.add_argument("filepath", help="Path to reproduction_state.yaml")

    args = parser.parse_args()

    if args.command == "init":
        result = init_state(
            task_id=args.task_id,
            pdf_path=args.pdf_path,
            arxiv_id=args.arxiv_id,
            paper_title=args.paper_title,
            output_dir=args.output_dir,
        )
        print(f"State file created: {args.output_dir}/reproduction_state.yaml")
        print(f"Status: {result['status']}")

    elif args.command == "read":
        result = read_state(args.filepath)
        if "success" in result and not result["success"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(1)
        if yaml:
            print(yaml.dump({k: v for k, v in result.items() if not k.startswith("_")},
                            default_flow_style=False, allow_unicode=True, sort_keys=False))
        else:
            print(json.dumps({k: v for k, v in result.items() if not k.startswith("_")},
                             ensure_ascii=False, indent=2))

    elif args.command == "update":
        updates = _parse_updates(args.set)
        if not updates:
            print("No updates specified. Use --set key=value")
            sys.exit(1)
        result = update_state(args.filepath, updates)
        if "success" in result and not result["success"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(1)
        print(f"Updated {args.filepath} with {len(updates)} field(s)")

    elif args.command == "validate":
        result = validate_state(args.filepath)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result.get("complete"):
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
