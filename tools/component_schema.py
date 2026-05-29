#!/usr/bin/env python3
"""
Component Schema & Registry

Defines the component.yaml specification and manages the .registry.yaml index
for searching, matching, and dependency resolution across the component library.

Schema specification (component.yaml):
    name: str               # Unique component identifier (snake_case)
    type: str               # Component type (see VALID_TYPES below)
    domain: str             # cv | nlp | llm | multimodal
    spec_version: str       # Schema version this component adheres to
    description: str        # One-line description of what it does
    source:                 # Provenance
      paper: str            # Paper title
      arxiv_id: str         # arXiv ID (optional)
      url: str              # Reference URL (optional)
    interface:              # Input/output specification
      inputs: dict          # {param_name: type_hint}
      outputs: dict         # {result_name: shape_hint}
    dependencies: list      # [{package: "torch>=1.8"}, ...]
    tags: list[str]         # Searchable keywords
    code_file: str          # Relative path to the Python implementation

Usage:
    # Validate a component.yaml
    python component_schema.py validate <component.yaml>

    # Search the registry
    python component_schema.py search --domain cv --type backbone.block

    # Register a new component
    python component_schema.py register <component.yaml> --component-dir <dir>

    # List all components
    python component_schema.py list [--domain cv] [--type backbone]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

# ═══════════════════════════════════════════════════════════════════════════
# Schema definition
# ═══════════════════════════════════════════════════════════════════════════

SPEC_VERSION = "1.0"

VALID_DOMAINS = {"cv", "nlp", "llm", "multimodal", "general"}

VALID_TYPES = {
    # CV
    "cv.backbone",        # Feature extractor trunk (ResNet, ViT, etc.)
    "cv.neck",            # Feature aggregation (FPN, BiFPN)
    "cv.head",            # Task-specific output layer
    "cv.loss",            # Loss function
    "cv.augmentation",    # Data augmentation
    "cv.attention",       # Attention mechanism
    "cv.block",           # Generic reusable building block
    "cv.normalization",   # Normalization layer
    "cv.activation",      # Activation function

    # NLP
    "nlp.backbone",       # Encoder/decoder trunk
    "nlp.tokenizer",      # Tokenizer utility
    "nlp.head",           # Task-specific head
    "nlp.loss",           # Loss function
    "nlp.attention",      # Attention variant
    "nlp.embedding",      # Positional/segment embeddings
    "nlp.normalization",  # Normalization layer

    # LLM
    "llm.weight_injection",  # LoRA, Adapter, Prefix tuning
    "llm.attention",         # FlashAttention, GQA, MQA
    "llm.parallelism",       # FSDP, TP, PP configs
    "llm.quantization",      # GPTQ, AWQ, NF4
    "llm.inference",         # Speculative decoding, KV cache
    "llm.loss",              # DPO, RLHF losses
    "llm.prompt",            # Chat template, prompt formatting

    # General
    "general.training",    # Training utilities (LR schedules, EMA)
    "general.evaluation",  # Evaluation metrics
    "general.data",        # Data processing utilities
}

# Fields required in every component.yaml
REQUIRED_FIELDS = [
    "name",
    "type",
    "domain",
    "spec_version",
    "description",
    "interface",
]

# Fields required in interface
REQUIRED_INTERFACE_FIELDS = ["inputs", "outputs"]

# Component name pattern: snake_case, alphanumeric + underscore
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$")


# ═══════════════════════════════════════════════════════════════════════════
# Component YAML template
# ═══════════════════════════════════════════════════════════════════════════

COMPONENT_TEMPLATE = """# Component: {name}
# Documentation: https://github.com/.../paper-reproduce-skill

name: "{name}"
type: "{comp_type}"
domain: "{domain}"
spec_version: "{spec_version}"
description: "{description}"

source:
  paper: ""
  arxiv_id: ""
  url: ""

interface:
  inputs: {{}}
  outputs: {{}}

dependencies:
  - package: "torch>=1.8"

tags: []

# The Python module that implements this component (relative to the domain dir)
code_file: "{name}.py"
"""


def generate_component_template(name: str, comp_type: str, domain: str,
                                description: str = "") -> str:
    """Generate a component.yaml template for a new component."""
    return COMPONENT_TEMPLATE.format(
        name=name,
        comp_type=comp_type,
        domain=domain,
        spec_version=SPEC_VERSION,
        description=description or f"TODO: describe {name}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_component(component: dict, check_code_exists: bool = False,
                       component_dir: str = "") -> dict:
    """
    Validate a component.yaml dict (L1: structural check).

    Returns:
        {valid: bool, errors: [str], warnings: [str]}
    """
    errors = []
    warnings = []

    # ── Required fields ──
    for field in REQUIRED_FIELDS:
        if field not in component or not component[field]:
            errors.append(f"Missing required field: '{field}'")

    # ── Name validation ──
    name = component.get("name", "")
    if name and not NAME_PATTERN.match(name):
        errors.append(
            f"Invalid component name '{name}': must be snake_case "
            f"(lowercase letters, digits, underscores)"
        )

    # ── Type validation ──
    comp_type = component.get("type", "")
    if comp_type and comp_type not in VALID_TYPES:
        similar = [t for t in VALID_TYPES if comp_type.split(".")[-1] in t]
        hint = f" Did you mean one of: {similar}" if similar else ""
        errors.append(f"Invalid type '{comp_type}'. Valid types: {sorted(VALID_TYPES)}.{hint}")

    # ── Domain validation ──
    domain = component.get("domain", "")
    if domain and domain not in VALID_DOMAINS:
        errors.append(f"Invalid domain '{domain}'. Valid domains: {sorted(VALID_DOMAINS)}")

    # ── Spec version ──
    spec = component.get("spec_version", "")
    if spec and spec != SPEC_VERSION:
        warnings.append(f"spec_version '{spec}' != current '{SPEC_VERSION}'; may need migration")

    # ── Interface validation ──
    interface = component.get("interface", {})
    if interface:
        for field in REQUIRED_INTERFACE_FIELDS:
            if field not in interface:
                errors.append(f"Missing interface field: '{field}'")
    else:
        errors.append("'interface' section is empty")

    # ── Dependencies ──
    deps = component.get("dependencies", [])
    for dep in deps:
        if not isinstance(dep, dict) or "package" not in dep:
            errors.append(f"Invalid dependency entry: {dep}. Must be {{package: 'name>=version'}}")

    # ── Code file existence (L2 pre-check) ──
    if check_code_exists and component_dir:
        code_file = component.get("code_file", f"{name}.py")
        code_path = os.path.join(component_dir, code_file)
        if not os.path.exists(code_path):
            warnings.append(f"code_file '{code_file}' not found at {component_dir}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Registry management
# ═══════════════════════════════════════════════════════════════════════════

REGISTRY_HEADER = """# Component Registry
# ==================
# Auto-generated index of all registered components.
# Updated: {timestamp}
# Total components: {count}
#
"""


def load_registry(registry_path: str) -> dict:
    """Load .registry.yaml, returning a dict. Creates empty if not found."""
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            return yaml.safe_load(f)
    return {"components": [], "updated": "", "total": 0}


def save_registry(registry: dict, registry_path: str):
    """Save registry dict to .registry.yaml."""
    registry["updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    registry["total"] = len(registry.get("components", []))

    with open(registry_path, "w") as f:
        f.write(REGISTRY_HEADER.format(
            timestamp=registry["updated"],
            count=registry["total"],
        ))
        yaml.dump(
            {"components": registry.get("components", [])},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def register_component(component_yaml_path: str, component_dir: str,
                       registry_path: str) -> dict:
    """
    Register a component into the registry.

    1. Validate the component.yaml
    2. Check for name conflicts
    3. Add to registry index
    4. Save updated registry
    """
    # Load component
    with open(component_yaml_path, "r") as f:
        component = yaml.safe_load(f) if yaml else json.load(f)

    # Validate
    validation = validate_component(component, check_code_exists=True,
                                    component_dir=component_dir)
    if not validation["valid"]:
        return {
            "success": False,
            "error": "Component validation failed",
            "validation": validation,
        }

    # Check for duplicates
    registry = load_registry(registry_path)
    existing_names = {c["name"] for c in registry.get("components", [])}
    if component["name"] in existing_names:
        return {
            "success": False,
            "error": f"Component '{component['name']}' already exists in registry",
        }

    # Build registry entry (subset of component.yaml for fast search)
    entry = {
        "name": component["name"],
        "type": component["type"],
        "domain": component["domain"],
        "description": component.get("description", ""),
        "tags": component.get("tags", []),
        "dependencies": [d.get("package", "") for d in component.get("dependencies", [])],
        "code_file": component.get("code_file", f"{component['name']}.py"),
        "source_paper": component.get("source", {}).get("paper", ""),
        "registered": datetime.now().strftime("%Y-%m-%d"),
    }

    registry.setdefault("components", []).append(entry)
    save_registry(registry, registry_path)

    return {
        "success": True,
        "message": f"Component '{component['name']}' registered successfully",
        "entry": entry,
    }


def search_registry(registry_path: str, domain: str = None, comp_type: str = None,
                    keyword: str = None) -> list:
    """
    Search the component registry by domain, type, and/or keyword.

    Returns a list of matching registry entries.
    """
    registry = load_registry(registry_path)
    components = registry.get("components", [])
    results = []

    for comp in components:
        # Domain filter
        if domain and comp.get("domain") != domain:
            continue
        # Type filter (partial match: "backbone" matches "cv.backbone")
        if comp_type and comp_type not in comp.get("type", ""):
            continue
        # Keyword search (in name, description, tags)
        if keyword:
            kw = keyword.lower()
            searchable = " ".join([
                comp.get("name", ""),
                comp.get("description", ""),
                " ".join(comp.get("tags", [])),
                comp.get("source_paper", ""),
            ]).lower()
            if kw not in searchable:
                continue
        results.append(comp)

    return results


def list_types() -> dict:
    """Return all valid component types organized by domain."""
    by_domain = {}
    for t in sorted(VALID_TYPES):
        domain = t.split(".")[0]
        by_domain.setdefault(domain, []).append(t)
    return by_domain


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Component schema validation and registry management"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # validate
    valid_parser = subparsers.add_parser("validate", help="Validate a component.yaml file")
    valid_parser.add_argument("component_yaml", help="Path to component.yaml")
    valid_parser.add_argument("--check-code", action="store_true",
                              help="Also check that code_file exists")
    valid_parser.add_argument("--component-dir", default=".",
                              help="Base directory for code file lookup")

    # search
    search_parser = subparsers.add_parser("search", help="Search the component registry")
    search_parser.add_argument("--registry", default=".registry.yaml",
                               help="Path to .registry.yaml")
    search_parser.add_argument("--domain", help="Filter by domain (cv/nlp/llm)")
    search_parser.add_argument("--type", help="Filter by component type")
    search_parser.add_argument("--keyword", "-k", help="Search keyword (name, tags, description)")

    # register
    reg_parser = subparsers.add_parser("register", help="Register a component")
    reg_parser.add_argument("component_yaml", help="Path to component.yaml")
    reg_parser.add_argument("--component-dir", required=True,
                            help="Directory containing the component code")
    reg_parser.add_argument("--registry", default=".registry.yaml",
                            help="Path to .registry.yaml")

    # list
    list_parser = subparsers.add_parser("list", help="List components or types")
    list_parser.add_argument("--registry", default=".registry.yaml",
                             help="Path to .registry.yaml")
    list_parser.add_argument("--domain", help="Filter by domain")
    list_parser.add_argument("--types", action="store_true",
                             help="List valid component types instead")

    # template
    tmpl_parser = subparsers.add_parser("template", help="Generate a component.yaml template")
    tmpl_parser.add_argument("name", help="Component name (snake_case)")
    tmpl_parser.add_argument("--type", required=True, help="Component type")
    tmpl_parser.add_argument("--domain", required=True, help="Component domain")
    tmpl_parser.add_argument("--description", default="", help="Component description")

    args = parser.parse_args()

    if args.command == "validate":
        with open(args.component_yaml, "r") as f:
            comp = yaml.safe_load(f) if yaml else json.load(f)
        result = validate_component(comp, args.check_code, args.component_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["valid"]:
            sys.exit(1)

    elif args.command == "search":
        results = search_registry(args.registry, args.domain, args.type, args.keyword)
        print(f"Found {len(results)} component(s):")
        for r in results:
            print(f"  [{r['domain']}/{r['type']}] {r['name']} — {r.get('description', '')}")

    elif args.command == "register":
        result = register_component(args.component_yaml, args.component_dir, args.registry)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["success"]:
            sys.exit(1)

    elif args.command == "list":
        if args.types:
            for domain, types in list_types().items():
                print(f"\n{domain}:")
                for t in types:
                    print(f"  - {t}")
        else:
            results = search_registry(args.registry, args.domain)
            if results:
                print(f"Components ({len(results)}):")
                for r in results:
                    print(f"  [{r['domain']}/{r['type']}] {r['name']}")
            else:
                print("Registry is empty. Use 'register' to add components.")

    elif args.command == "template":
        print(generate_component_template(args.name, args.type, args.domain, args.description))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
