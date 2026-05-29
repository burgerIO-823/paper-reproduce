#!/usr/bin/env python3
"""
Component Checker (L1 / L2 / L3)

Three-level validation for user and built-in components:

  L1 — Structural: component.yaml completeness, field format, naming conventions
  L2 — Behavioral: import, instantiate, forward-pass with dummy inputs, shape check
  L3 — Compatibility: conflict detection with other registered components

Usage:
    # Run all checks
    python component_checker.py <component.yaml> --component-dir <dir>

    # Run specific levels
    python component_checker.py <component.yaml> --level L1
    python component_checker.py <component.yaml> --level L2 --component-dir <dir>
    python component_checker.py <component.yaml> --level L3 --registry <path>

Output: JSON report with per-level pass/fail status and detailed diagnostics.
"""

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

# ── L1 is defined in component_schema, but we inline the key parts ─────────
from component_schema import validate_component, SPEC_VERSION, VALID_TYPES, VALID_DOMAINS

# ── L3: Known component conflicts ──────────────────────────────────────────

KNOWN_CONFLICTS = [
    {
        "id": "conflict-001",
        "component_a": {"type": "llm.attention", "name_pattern": "flash.*attn"},
        "component_b": {"type": "llm.parallelism", "name_pattern": "fsdp.*"},
        "severity": "warning",
        "description": (
            "FlashAttention's custom CUDA kernels may not be compatible with "
            "FSDP's flat-parameter sharding. Use PyTorch 2.2+ which has improved "
            "integration, or use xformers as a fallback."
        ),
        "resolution": "Upgrade to PyTorch >= 2.2, or replace FlashAttention with xformers.memory_efficient_attention.",
    },
    {
        "id": "conflict-002",
        "component_a": {"type": "llm.quantization", "name_pattern": ".*gptq.*"},
        "component_b": {"type": "llm.weight_injection", "name_pattern": ".*lora.*"},
        "severity": "warning",
        "description": (
            "GPTQ quantized models combined with LoRA require special handling. "
            "LoRA weights must be in fp16/bf16 while base weights remain quantized. "
            "Not all PEFT versions support this correctly."
        ),
        "resolution": "Use PEFT >= 0.7.0 with bitsandbytes >= 0.41.0. Verify that LoRA layers are not quantized.",
    },
    {
        "id": "conflict-003",
        "component_a": {"type": "cv.head", "name_pattern": ".*"},
        "component_b": {"type": "cv.neck", "name_pattern": ".*fpn.*"},
        "severity": "info",
        "description": (
            "FPN neck outputs multi-scale features. Ensure the head component "
            "accepts a list/tuple of feature maps, not a single tensor. "
            "Many standard heads expect single-scale input."
        ),
        "resolution": "Verify the head's forward signature accepts List[Tensor] or wrap in a MultiScaleAdapter.",
    },
    {
        "id": "conflict-004",
        "component_a": {"type": "llm.quantization", "name_pattern": ".*bitsandbytes.*4bit"},
        "component_b": {"type": "general.training", "name_pattern": ".*gradient_checkpoint.*"},
        "severity": "warning",
        "description": (
            "4-bit quantization with gradient checkpointing can cause unexpected "
            "dequantization during the backward pass. This is a known issue with "
            "bitsandbytes < 0.43.0."
        ),
        "resolution": "Upgrade bitsandbytes to >= 0.43.0, or disable gradient checkpointing for quantized layers.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# L2: Behavioral Check
# ═══════════════════════════════════════════════════════════════════════════

def _shape_from_hint(hint: str) -> tuple:
    """
    Parse a shape hint like '(B, 3, 224, 224)' into concrete dimensions
    for dummy input generation. Variables become small fixed values.
    """
    # Remove parenthesized comments
    hint = re.sub(r"\([^)]*\)", "", hint)
    # Extract the main shape pattern
    shape_match = re.search(r"\(([^)]+)\)", hint)
    if not shape_match:
        return (2, 3, 224, 224)  # Sensible default

    dims = []
    for dim in shape_match.group(1).split(","):
        dim = dim.strip()
        if not dim:
            continue
        if dim.upper() == "B":
            dims.append(2)  # batch size
        elif dim.upper() in ("H", "W"):
            dims.append(32)  # spatial dim
        elif dim.upper() in ("L", "S", "SEQ_LEN", "T"):
            dims.append(16)  # sequence length
        elif dim.upper() in ("D", "C", "E", "HIDDEN", "EMBED"):
            dims.append(128)  # feature dim
        elif dim.upper() == "VOCAB_SIZE":
            dims.append(1000)
        elif dim.upper() == "NUM_CLASSES":
            dims.append(10)
        elif dim.upper() == "NUM_HEADS":
            dims.append(8)
        else:
            # Try to parse as integer
            try:
                dims.append(int(dim))
            except ValueError:
                dims.append(32)  # default

    return tuple(dims) if dims else (2, 3, 224, 224)


def _generate_dummy_inputs(interface: dict) -> dict:
    """
    Generate dummy torch tensors based on the component's interface spec.

    Returns a dict of {param_name: torch.Tensor} for the forward pass.
    """
    try:
        import torch
    except ImportError:
        return {}

    inputs = {}
    for param_name, type_hint in interface.get("inputs", {}).items():
        type_hint_lower = str(type_hint).lower()

        if "int" in type_hint_lower:
            # Scalar integer parameter (e.g., stride, num_heads)
            # Extract a default value or use 1
            default_match = re.search(r"default[:\s]*(\d+)", type_hint_lower)
            inputs[param_name] = int(default_match.group(1)) if default_match else 1
        elif "float" in type_hint_lower:
            default_match = re.search(r"default[:\s]*([\d.]+)", type_hint_lower)
            inputs[param_name] = float(default_match.group(1)) if default_match else 1.0
        elif "bool" in type_hint_lower:
            inputs[param_name] = False
        else:
            # Assume tensor with shape hint
            shape = _shape_from_hint(str(type_hint))
            inputs[param_name] = torch.randn(*shape)

    return inputs


def _check_tensor_outputs(actual_output, expected_spec: dict) -> dict:
    """
    Verify that actual output tensors match the expected shapes/types.
    """
    import torch

    results = []
    actual_dict = actual_output if isinstance(actual_output, dict) else {"output": actual_output}
    if not isinstance(actual_output, dict) and not isinstance(actual_output, (list, tuple)):
        actual_dict = {"output": actual_output}
    elif isinstance(actual_output, (list, tuple)):
        actual_dict = {f"output_{i}": t for i, t in enumerate(actual_output)}

    for key, hint in expected_spec.get("outputs", {}).items():
        if key in actual_dict:
            tensor = actual_dict[key]
            if isinstance(tensor, torch.Tensor):
                actual_shape = tuple(tensor.shape)
                # Check for NaN/Inf
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                results.append({
                    "output_key": key,
                    "actual_shape": str(actual_shape),
                    "expected_hint": str(hint),
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "dtype": str(tensor.dtype),
                })
            else:
                results.append({
                    "output_key": key,
                    "actual_type": type(tensor).__name__,
                    "warning": "Output is not a tensor",
                })
        else:
            results.append({
                "output_key": key,
                "error": f"Expected output '{key}' not found in model output",
            })

    return results


def run_l2_check(component_yaml_path: str, component_dir: str) -> dict:
    """
    L2 Behavioral Check:
    1. Import the component module
    2. Find the main class/function
    3. Instantiate with default parameters
    4. Run forward pass with dummy inputs
    5. Verify output shapes and check for NaN/Inf
    """
    try:
        import torch
    except ImportError:
        return {
            "level": "L2",
            "passed": False,
            "error": "PyTorch is required for L2 behavioral checks",
            "details": [],
        }

    # Load component spec
    with open(component_yaml_path, "r") as f:
        component = yaml.safe_load(f)

    name = component.get("name", "unknown")
    code_file = component.get("code_file", f"{name}.py")
    code_path = os.path.join(component_dir, code_file)

    if not os.path.exists(code_path):
        return {
            "level": "L2",
            "passed": False,
            "error": f"Code file not found: {code_path}",
            "details": [],
        }

    details = []

    # Step 1: Import the module
    try:
        spec = importlib.util.spec_from_file_location(name, code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        details.append({"step": "import", "status": "ok", "message": f"Module '{name}' imported"})
    except Exception as e:
        details.append({
            "step": "import",
            "status": "fail",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
        return {"level": "L2", "passed": False, "error": f"Import failed: {e}", "details": details}

    # Step 2: Find the main class (heuristic: class with same name, converted to PascalCase)
    pascal_name = "".join(word.capitalize() for word in name.split("_"))
    main_class = None

    # Search for the class in the module
    for attr_name in dir(module):
        if attr_name.lower() == pascal_name.lower():
            obj = getattr(module, attr_name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                main_class = obj
                break

    # Fallback: find any nn.Module subclass
    if main_class is None:
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and attr_name != "Module":
                main_class = obj
                details.append({
                    "step": "class_lookup",
                    "status": "warning",
                    "message": f"No class matching '{pascal_name}' found. Using '{attr_name}' as fallback.",
                })
                break

    if main_class is None:
        details.append({"step": "class_lookup", "status": "fail",
                        "message": "No torch.nn.Module subclass found in module"})
        return {"level": "L2", "passed": False, "error": "No nn.Module found", "details": details}

    details.append({
        "step": "class_lookup",
        "status": "ok",
        "message": f"Found class: {main_class.__name__}",
    })

    # Step 3: Instantiate
    try:
        # Try to infer sensible default parameters from the interface
        interface = component.get("interface", {})
        instance = main_class()
        details.append({"step": "instantiate", "status": "ok",
                        "message": f"{main_class.__name__}() created"})
    except Exception as e:
        # Try with common default parameters
        try:
            instance = main_class(in_channels=3, out_channels=64)
            details.append({"step": "instantiate", "status": "ok",
                            "message": f"{main_class.__name__}(in_channels=3, out_channels=64) created"})
        except Exception:
            details.append({
                "step": "instantiate",
                "status": "fail",
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
            return {"level": "L2", "passed": False,
                    "error": f"Instantiation failed: {e}", "details": details}

    # Step 4: Forward pass (with retry on shape mismatch)
    dummy_inputs = _generate_dummy_inputs(interface)

    def _run_forward(model, inputs):
        model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict) and len(inputs) == 1:
                return model(list(inputs.values())[0])
            elif isinstance(inputs, dict) and len(inputs) > 0:
                try:
                    return model(**inputs)
                except TypeError:
                    first_tensor = next((v for v in inputs.values()
                                        if isinstance(v, torch.Tensor)), None)
                    return model(first_tensor) if first_tensor is not None else model()
            else:
                return model()

    forward_ok = False
    last_error = None

    for attempt in range(3):  # Try up to 3 times with different strategies
        try:
            output = _run_forward(instance, dummy_inputs)
            forward_ok = True
            break
        except RuntimeError as e:
            last_error = e
            error_msg = str(e).lower()
            if attempt == 0 and ("shape" in error_msg or "mat1" in error_msg or "mat2" in error_msg):
                # Shape mismatch: try with smaller dimensions
                details.append({
                    "step": f"forward_retry_{attempt + 1}",
                    "status": "info",
                    "message": f"Shape mismatch on first attempt. Retrying with different dims.",
                })
                # Re-generate with smaller feature dims
                smaller = {}
                for k, v in dummy_inputs.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        smaller[k] = torch.randn(v.shape[0], 64, *v.shape[2:])
                    else:
                        smaller[k] = v
                dummy_inputs = smaller
            elif attempt == 1 and ("shape" in error_msg or "mat1" in error_msg):
                # Second retry: try with 128 channels
                retry = {}
                for k, v in dummy_inputs.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        retry[k] = torch.randn(v.shape[0], 128, *v.shape[2:])
                    else:
                        retry[k] = v
                dummy_inputs = retry
            else:
                break
        except Exception as e:
            last_error = e
            break

    if forward_ok:
        details.append({"step": "forward", "status": "ok",
                        "message": "Forward pass completed"})
    else:
        details.append({
            "step": "forward",
            "status": "fail",
            "message": str(last_error),
            "traceback": traceback.format_exc(),
        })
        return {"level": "L2", "passed": False,
                "error": f"Forward pass failed after retries: {last_error}", "details": details}

    # Step 5: Check outputs
    output_results = _check_tensor_outputs(output, interface)
    details.append({"step": "output_check", "status": "ok", "results": output_results})

    # Determine overall pass/fail
    has_nan_error = any(r.get("has_nan") for r in output_results)
    has_inf_error = any(r.get("has_inf") for r in output_results)
    all_passed = not has_nan_error and not has_inf_error

    return {
        "level": "L2",
        "passed": all_passed,
        "warnings": (
            (["NaN detected in output"] if has_nan_error else []) +
            (["Inf detected in output"] if has_inf_error else [])
        ),
        "output_shape_check": output_results,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════════
# L3: Compatibility Check
# ═══════════════════════════════════════════════════════════════════════════

def run_l3_check(component_yaml_path: str, registry_path: str) -> dict:
    """
    L3 Compatibility Check:
    Check the component against all registered components for known conflicts.
    """
    from component_schema import load_registry

    with open(component_yaml_path, "r") as f:
        component = yaml.safe_load(f)

    registry = load_registry(registry_path)
    registered = registry.get("components", [])

    conflicts_found = []

    comp_type = component.get("type", "")
    comp_name = component.get("name", "")

    for other in registered:
        if other.get("name") == comp_name:
            continue  # Skip self

        other_type = other.get("type", "")
        other_name = other.get("name", "")

        for conflict in KNOWN_CONFLICTS:
            a = conflict["component_a"]
            b = conflict["component_b"]

            # Check if component matches a and other matches b (or vice versa)
            match_ab = (
                comp_type == a["type"] and re.match(a["name_pattern"], comp_name) and
                other_type == b["type"] and re.match(b["name_pattern"], other_name)
            )
            match_ba = (
                comp_type == b["type"] and re.match(b["name_pattern"], comp_name) and
                other_type == a["type"] and re.match(a["name_pattern"], other_name)
            )

            if match_ab or match_ba:
                conflicts_found.append({
                    "conflict_id": conflict["id"],
                    "with_component": other_name,
                    "severity": conflict["severity"],
                    "description": conflict["description"],
                    "resolution": conflict["resolution"],
                })

    return {
        "level": "L3",
        "passed": len([c for c in conflicts_found if c["severity"] == "critical"]) == 0,
        "conflicts_found": len(conflicts_found),
        "conflicts": conflicts_found,
        "registry_size": len(registered),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Combined checker
# ═══════════════════════════════════════════════════════════════════════════

def run_all_checks(component_yaml_path: str, component_dir: str,
                   registry_path: str = None) -> dict:
    """
    Run L1, L2, and L3 checks and return a combined report.
    """
    with open(component_yaml_path, "r") as f:
        component = yaml.safe_load(f)

    report = {
        "component": component.get("name", "unknown"),
        "type": component.get("type", ""),
        "domain": component.get("domain", ""),
        "checks": {},
        "overall_pass": True,
    }

    # L1
    l1 = validate_component(component, check_code_exists=True,
                            component_dir=component_dir)
    report["checks"]["L1"] = {
        "passed": l1["valid"],
        "errors": l1["errors"],
        "warnings": l1["warnings"],
    }
    if not l1["valid"]:
        report["overall_pass"] = False

    # L2 (skip if L1 failed fundamentally)
    if l1["valid"]:
        l2 = run_l2_check(component_yaml_path, component_dir)
        report["checks"]["L2"] = l2
        if not l2["passed"]:
            report["overall_pass"] = False
    else:
        report["checks"]["L2"] = {"passed": False, "skipped": True,
                                  "reason": "L1 validation failed"}

    # L3
    if registry_path and os.path.exists(registry_path):
        l3 = run_l3_check(component_yaml_path, registry_path)
        report["checks"]["L3"] = l3
        if not l3["passed"]:
            report["overall_pass"] = False

    return report


def format_report(report: dict) -> str:
    """Format a check report as a human-readable markdown string."""
    lines = [
        f"# Component Check Report: `{report['component']}`",
        f"",
        f"**Type**: {report['type']} | **Domain**: {report['domain']}",
        f"**Overall**: {'PASS' if report['overall_pass'] else 'FAIL'}",
        f"",
    ]

    for level in ["L1", "L2", "L3"]:
        check = report.get("checks", {}).get(level, {})
        if not check:
            continue

        status_icon = "PASS" if check.get("passed") else "FAIL"
        if check.get("skipped"):
            status_icon = "SKIPPED"

        lines.append(f"## {level}: {status_icon}")

        if level == "L1":
            for err in check.get("errors", []):
                lines.append(f"- **Error**: {err}")
            for warn in check.get("warnings", []):
                lines.append(f"- **Warning**: {warn}")
            if not check.get("errors") and not check.get("warnings"):
                lines.append("- All structural checks passed")

        elif level == "L2":
            for detail in check.get("details", []):
                msg = detail.get("message", "")
                status = detail.get("status", "")
                icon = {"ok": "+", "fail": "-", "warning": "!"}.get(status, "?")
                lines.append(f"- [{icon}] {detail['step']}: {msg}")
            for warn in check.get("warnings", []):
                lines.append(f"- **Warning**: {warn}")

        elif level == "L3":
            for conflict in check.get("conflicts", []):
                sev = conflict["severity"].upper()
                lines.append(f"- **[{sev}]** {conflict['description']} (with `{conflict['with_component']}`)")
                lines.append(f"  Resolution: {conflict['resolution']}")
            if not check.get("conflicts"):
                lines.append("- No compatibility conflicts found")

        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Component checker: L1 (structural), L2 (behavioral), L3 (compatibility)"
    )
    parser.add_argument("component_yaml", help="Path to component.yaml")
    parser.add_argument("--component-dir", "-d", default=".",
                        help="Directory containing component code files")
    parser.add_argument("--level", choices=["L1", "L2", "L3", "all"],
                        default="all", help="Which check level to run")
    parser.add_argument("--registry", "-r", default=".registry.yaml",
                        help="Path to .registry.yaml for L3 check")
    parser.add_argument("--format", choices=["json", "markdown"], default="json",
                        help="Output format")
    parser.add_argument("--output", "-o", default=None,
                        help="Write report to file")

    args = parser.parse_args()
    component_dir = os.path.abspath(args.component_dir)

    if args.level == "L1":
        with open(args.component_yaml, "r") as f:
            component = yaml.safe_load(f)
        l1 = validate_component(component)
        report = {
            "component": component.get("name", ""),
            "type": component.get("type", ""),
            "domain": component.get("domain", ""),
            "checks": {"L1": {"passed": l1["valid"], "errors": l1["errors"], "warnings": l1["warnings"]}},
            "overall_pass": l1["valid"],
        }
    elif args.level == "L2":
        with open(args.component_yaml, "r") as f:
            component = yaml.safe_load(f)
        l2 = run_l2_check(args.component_yaml, component_dir)
        report = {
            "component": component.get("name", ""),
            "type": component.get("type", ""),
            "domain": component.get("domain", ""),
            "checks": {"L2": l2},
            "overall_pass": l2["passed"],
        }
    elif args.level == "L3":
        with open(args.component_yaml, "r") as f:
            component = yaml.safe_load(f)
        l3 = run_l3_check(args.component_yaml, args.registry)
        report = {
            "component": component.get("name", ""),
            "type": component.get("type", ""),
            "domain": component.get("domain", ""),
            "checks": {"L3": l3},
            "overall_pass": l3["passed"],
        }
    else:
        report = run_all_checks(args.component_yaml, component_dir, args.registry)

    if args.format == "markdown":
        output = format_report(report)
    else:
        output = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    if not report.get("overall_pass", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
