#!/usr/bin/env python3
"""
Component Normalizer

Helps users register custom components by:
  1. Extracting code structure (classes, signatures, imports)
  2. Generating an "intent recognition" prompt for Claude to analyze
  3. Applying mechanical normalizations (naming, interface wrapping, component.yaml)
  4. Showing a diff for user approval
  5. Registering the normalized component

Usage:
    # Step 1: Analyze a user component and generate intent prompt
    python component_normalizer.py analyze <user_code.py> [--domain cv]

    # Step 2: Apply normalization with Claude's analysis
    python component_normalizer.py normalize <user_code.py> \\
        --intent <intent_json> [--output-dir <dir>] [--dry-run]

    # Step 3: Register the normalized component
    (handled by component_schema.py register)
"""

import argparse
import ast
import difflib
import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

from component_schema import (
    SPEC_VERSION, VALID_TYPES, VALID_DOMAINS,
    validate_component, generate_component_template,
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Code structure extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_code_structure(filepath: str) -> dict:
    """
    Parse a Python file and extract its structure:
      - module docstring
      - classes (name, base classes, methods with signatures)
      - functions (name, signature, docstring)
      - imports
      - top-level variables (for hyperparameters)
    """
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    structure = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "docstring": ast.get_docstring(tree) or "",
        "classes": [],
        "functions": [],
        "imports": [],
        "hyperparams": [],
        "total_lines": len(source.split("\n")),
    }

    for node in ast.iter_child_nodes(tree):
        # Classes
        if isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "bases": [_name_of(n) for n in node.bases],
                "docstring": ast.get_docstring(node) or "",
                "methods": [],
            }
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    class_info["methods"].append({
                        "name": item.name,
                        "args": [_arg_to_str(a) for a in item.args.args],
                        "defaults": len(item.args.defaults),
                        "docstring": ast.get_docstring(item) or "",
                        "is_forward": item.name == "forward",
                    })
            structure["classes"].append(class_info)

        # Standalone functions
        elif isinstance(node, ast.FunctionDef):
            structure["functions"].append({
                "name": node.name,
                "args": [_arg_to_str(a) for a in node.args.args],
                "docstring": ast.get_docstring(node) or "",
            })

        # Imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                structure["imports"].append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(a.name for a in node.names)
            structure["imports"].append(f"from {node.module} import {names}")

        # Top-level assignments (potential hyperparams)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    structure["hyperparams"].append(target.id)

    return structure


def _name_of(node) -> str:
    """Get the string name of an AST name/attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    return "?"


def _arg_to_str(arg) -> str:
    """Convert an AST arg node to a string representation."""
    s = arg.arg
    if arg.annotation:
        s += f": {ast.unparse(arg.annotation)}"
    return s


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Intent recognition prompt generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_intent_prompt(structure: dict, domain: str = "cv") -> str:
    """
    Generate a structured prompt for Claude to identify:
      1. What type of component this is
      2. What its interface should be
      3. What needs to change to meet the spec
    """

    classes = structure.get("classes", [])
    functions = structure.get("functions", [])
    imports = structure.get("imports", [])

    main_class = classes[0]["name"] if classes else "Unknown"
    bases = classes[0]["bases"] if classes else []
    is_nn_module = any("Module" in b or "nn.Module" in b for b in bases)

    # Detect likely domain from imports
    detected_domain = domain
    import_text = " ".join(imports).lower()
    if "transformers" in import_text or "tokenizer" in import_text:
        detected_domain = "nlp"
    if "peft" in import_text or "lora" in import_text or "llama" in import_text:
        detected_domain = "llm"
    if "torchvision" in import_text or "timm" in import_text:
        detected_domain = "cv"

    lines = [
        "# Component Intent Recognition",
        "",
        "Analyze the following user-submitted component code and determine its "
        "correct classification and interface specification.",
        "",
        "## Code Structure",
        f"- **File**: {structure['filename']} ({structure['total_lines']} lines)",
        f"- **Main class**: `{main_class}`",
        f"- **Base classes**: {bases if bases else 'none'}",
        f"- **Is nn.Module**: {is_nn_module}",
        f"- **Detected domain (from imports)**: {detected_domain}",
        "",
    ]

    if classes:
        lines.append("### Classes")
        for cls in classes:
            lines.append(f"#### `{cls['name']}`({', '.join(cls['bases'])})")
            if cls["docstring"]:
                lines.append(f"> {cls['docstring']}")
            lines.append("")
            lines.append("**Methods:**")
            for m in cls["methods"]:
                args_str = ", ".join(m["args"])
                lines.append(f"- `{m['name']}({args_str})`")
                if m["docstring"]:
                    lines.append(f"  > {m['docstring']}")
            lines.append("")

    if functions:
        lines.append("### Standalone Functions")
        for fn in functions:
            args_str = ", ".join(fn["args"])
            lines.append(f"- `{fn['name']}({args_str})`")
            if fn["docstring"]:
                lines.append(f"  > {fn['docstring']}")
        lines.append("")

    if imports:
        lines.append("### Imports")
        for imp in imports:
            lines.append(f"- `{imp}`")
        lines.append("")

    lines.append("## Valid Component Types")
    lines.append("")
    type_lines = []
    for t in sorted(VALID_TYPES):
        domain_part = t.split(".")[0]
        if domain_part in (detected_domain, "general"):
            type_lines.append(f"- `{t}`")
    if not type_lines:
        type_lines = [f"- `{t}`" for t in sorted(VALID_TYPES)]
    lines.extend(type_lines)
    lines.append("")

    lines.append("## Instructions")
    lines.append("")
    lines.append("Based on the code structure above, fill in the following YAML with your analysis:")
    lines.append("")
    lines.append("```yaml")
    lines.append("intent:")
    lines.append(f"  suggested_name: \"{_suggest_name(structure['filename'])}\"  # snake_case component name")
    lines.append("  type: \"cv.backbone\"          # one of the valid types above")
    lines.append(f"  domain: \"{detected_domain}\"")
    lines.append("  description: \"...\"           # one-line description of what this component does")
    lines.append("")
    lines.append("  interface:")
    lines.append("    inputs:")
    lines.append("      x: \"(B, C, H, W)\"       # input parameter: shape hint")
    lines.append("    outputs:")
    lines.append("      output: \"(B, C, H, W)\"   # output: shape hint")
    lines.append("")
    lines.append("  # What needs to change for standardization?")
    lines.append("  normalizations:")
    lines.append("    - type: \"rename_class\"")
    lines.append("      from: \"\"                # current name")
    lines.append("      to: \"\"                  # suggested name (PascalCase)")
    lines.append("    - type: \"add_config_param\"")
    lines.append("      param: \"\"               # hardcoded value that should be a parameter")
    lines.append("      current_value: \"\"")
    lines.append("      suggested_default: \"\"")
    lines.append("    - type: \"adapt_interface\"")
    lines.append("      description: \"\"         # how the interface should be adapted")
    lines.append("")
    lines.append("  dependencies:")
    lines.append("    - package: \"torch>=1.8\"")
    lines.append("")
    lines.append("  tags: []")
    lines.append("```")

    return "\n".join(lines)


def _suggest_name(filename: str) -> str:
    """Suggest a snake_case component name from a filename."""
    name = os.path.splitext(filename)[0]
    # Convert CamelCase to snake_case
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return name.lower().replace("-", "_")


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Normalization
# ═══════════════════════════════════════════════════════════════════════════

def apply_normalization(source_file: str, intent: dict, output_dir: str,
                        dry_run: bool = False) -> dict:
    """
    Apply mechanical normalizations based on Claude's intent analysis.

    Produces:
      - Normalized .py file in output_dir
      - component.yaml in output_dir
      - Diff between original and normalized
    """
    suggested_name = intent.get("suggested_name", "")
    norm_list = intent.get("normalizations", [])
    interface = intent.get("interface", {"inputs": {}, "outputs": {}})
    comp_type = intent.get("type", "")
    domain = intent.get("domain", "cv")
    description = intent.get("description", "")
    deps = intent.get("dependencies", [{"package": "torch>=1.8"}])
    tags = intent.get("tags", [])

    os.makedirs(output_dir, exist_ok=True)

    # Read original source
    with open(source_file, "r") as f:
        original_source = f.read()

    normalized_source = original_source

    # Apply renames
    for norm in norm_list:
        if norm.get("type") == "rename_class":
            old_name = norm.get("from", "")
            new_name = norm.get("to", "")
            if old_name and new_name and old_name != new_name:
                normalized_source = re.sub(
                    rf"\b{re.escape(old_name)}\b", new_name, normalized_source
                )

        elif norm.get("type") == "add_config_param":
            # Add as __init__ parameter comment — actual code change is for Claude
            param = norm.get("param", "")
            current = norm.get("current_value", "")
            default = norm.get("suggested_default", "")
            note = (
                f"\n# NOTE: Parameter '{param}' was hardcoded as {current}. "
                f"Consider making it a constructor argument with default {default}.\n"
            )
            if note not in normalized_source:
                normalized_source += note

    # Write normalized code
    normalized_filename = f"{suggested_name}.py" if suggested_name else os.path.basename(source_file)
    normalized_path = os.path.join(output_dir, normalized_filename)

    if not dry_run:
        with open(normalized_path, "w") as f:
            f.write(normalized_source)

    # Generate component.yaml
    comp_yaml = {
        "name": suggested_name,
        "type": comp_type,
        "domain": domain,
        "spec_version": SPEC_VERSION,
        "description": description,
        "source": {
            "paper": "",
            "arxiv_id": "",
            "url": "",
        },
        "interface": interface,
        "dependencies": deps,
        "tags": tags,
        "code_file": normalized_filename,
    }

    comp_yaml_path = os.path.join(output_dir, f"{suggested_name}.yaml")

    if not dry_run:
        with open(comp_yaml_path, "w") as f:
            yaml.dump(comp_yaml, f, default_flow_style=False, allow_unicode=True)

    # Generate unified diff
    diff = "\n".join(difflib.unified_diff(
        original_source.split("\n"),
        normalized_source.split("\n"),
        fromfile=f"a/{os.path.basename(source_file)}",
        tofile=f"b/{normalized_filename}",
        lineterm="",
    ))

    return {
        "normalized_code": normalized_path if not dry_run else "(dry-run)",
        "component_yaml": comp_yaml_path if not dry_run else "(dry-run)",
        "diff": diff,
        "changes_summary": _summarize_changes(norm_list),
        "suggested_name": suggested_name,
        "component_yaml_content": comp_yaml,
    }


def _summarize_changes(normalizations: list) -> list:
    """Create a human-readable summary of all normalizations."""
    summaries = []
    for norm in normalizations:
        t = norm.get("type", "unknown")
        if t == "rename_class":
            summaries.append(f"Rename class: {norm.get('from', '?')} → {norm.get('to', '?')}")
        elif t == "add_config_param":
            summaries.append(
                f"Parameterize '{norm.get('param', '?')}': "
                f"was {norm.get('current_value', '?')}, "
                f"suggested default {norm.get('suggested_default', '?')}"
            )
        elif t == "adapt_interface":
            summaries.append(f"Interface adaptation: {norm.get('description', 'no details')}")
        else:
            summaries.append(f"Unknown normalization: {t}")
    return summaries


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Normalize user components for the component library"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Extract code structure and generate intent recognition prompt"
    )
    analyze_parser.add_argument("source_file", help="Path to the user's Python component")
    analyze_parser.add_argument("--domain", default="cv",
                                help="Likely domain (cv/nlp/llm)")
    analyze_parser.add_argument("--output", "-o", default=None,
                                help="Save the intent prompt to a file")

    # normalize
    norm_parser = subparsers.add_parser(
        "normalize",
        help="Apply normalizations based on intent analysis"
    )
    norm_parser.add_argument("source_file", help="Path to the user's Python component")
    norm_parser.add_argument("--intent", required=True,
                             help="Path to intent JSON/YAML (Claude's analysis output)")
    norm_parser.add_argument("--output-dir", "-o", default=".",
                             help="Output directory for normalized files")
    norm_parser.add_argument("--dry-run", action="store_true",
                             help="Show what would change without writing files")

    # diff-only
    diff_parser = subparsers.add_parser(
        "diff",
        help="Show diff of normalization without applying it"
    )
    diff_parser.add_argument("source_file", help="Path to the user's Python component")
    diff_parser.add_argument("--intent", required=True,
                             help="Path to intent JSON/YAML")

    args = parser.parse_args()

    if args.command == "analyze":
        structure = extract_code_structure(args.source_file)
        prompt = generate_intent_prompt(structure, args.domain)

        if args.output:
            with open(args.output, "w") as f:
                f.write(prompt)
            print(f"Intent prompt saved to {args.output}")
            # Also output the structure as JSON for tool chaining
            struct_path = args.output.replace(".md", "_structure.json")
            with open(struct_path, "w") as f:
                json.dump(structure, f, ensure_ascii=False, indent=2)
            print(f"Code structure saved to {struct_path}")
        else:
            print(prompt)

    elif args.command in ("normalize", "diff"):
        # Load intent
        with open(args.intent, "r") as f:
            intent_raw = f.read()

        # Try YAML first, then JSON
        if yaml:
            intent_data = yaml.safe_load(intent_raw)
        else:
            intent_data = json.loads(intent_raw)

        intent = intent_data.get("intent", intent_data)

        if args.command == "normalize":
            result = apply_normalization(
                args.source_file, intent, args.output_dir, args.dry_run
            )
            print(f"Component: {result['suggested_name']}")
            print(f"Output dir: {args.output_dir}")
            print(f"\nChanges:")
            for s in result["changes_summary"]:
                print(f"  - {s}")
            print(f"\nDiff:")
            print(result["diff"])

            if not args.dry_run:
                print(f"\nFiles written:")
                print(f"  Code: {result['normalized_code']}")
                print(f"  YAML: {result['component_yaml']}")
                print(f"\nTo validate: python component_checker.py {result['component_yaml']} --component-dir {args.output_dir}")
                print(f"To register: python component_schema.py register {result['component_yaml']} --component-dir {args.output_dir}")

        elif args.command == "diff":
            result = apply_normalization(args.source_file, intent, "/tmp", dry_run=True)
            print(result["diff"])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
