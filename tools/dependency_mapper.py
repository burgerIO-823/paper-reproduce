#!/usr/bin/env python3
"""
Dependency Mapper (Hybrid)

Maps parsed paper architecture and training recipe to concrete software
dependencies using a two-tier approach:

  Tier 1 (Rule-based): Deterministic mapping for well-known components
    - Backbone → library + pretrained weight source
    - Named components → required packages
    - Training features → supporting libraries

  Tier 2 (LLM reasoning): For novel, ambiguous, or unmatched components,
    generates a structured prompt for Claude to fill in the gaps.

The output separates confident rule-based results from items that need
LLM reasoning, enabling a human-in-the-loop or automated Claude review.

Usage:
    python dependency_mapper.py <reproduction_state.yaml> [--llm-prompt]
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: Rule-based knowledge bases
# ═══════════════════════════════════════════════════════════════════════════

# ── Backbone → Source mapping ──────────────────────────────────────────────

BACKBONE_SOURCES = [
    # CV - Classification
    (re.compile(r"ResNet[-\s]?(\d+)", re.IGNORECASE),
     "torchvision", "ResNet{0}_Weights.IMAGENET1K_V2", 0.95),
    (re.compile(r"ResNeXt[-\s]?(\d+)", re.IGNORECASE),
     "torchvision", "ResNeXt{0}_Weights.IMAGENET1K_V2", 0.95),
    (re.compile(r"ViT[-\s]?(?:B|L|H)?[-\s]?/?(\d+)", re.IGNORECASE),
     "timm", "vit_{0}_patch16_224", 0.90),
    (re.compile(r"Swin[-\s]?(?:T|S|B|L)?", re.IGNORECASE),
     "timm", None, 0.90),
    (re.compile(r"ConvNeXt[-\s]?(?:T|S|B|L|X)?", re.IGNORECASE),
     "timm", None, 0.90),
    (re.compile(r"EfficientNet[-\s]?(?:B|V)?(\d+)", re.IGNORECASE),
     "timm", "efficientnet_b{0}", 0.90),
    (re.compile(r"MobileNet[-\s]?V?(\d+)?", re.IGNORECASE),
     "torchvision", None, 0.85),
    (re.compile(r"DenseNet[-\s]?(\d+)", re.IGNORECASE),
     "torchvision", "DenseNet{0}_Weights.IMAGENET1K_V1", 0.95),
    # CV - Detection/Segmentation
    (re.compile(r"YOLO[-\s]?v?(\d+)", re.IGNORECASE),
     "ultralytics", None, 0.85),
    (re.compile(r"FPN", re.IGNORECASE),
     "torchvision.ops", None, 0.80),
    # NLP
    (re.compile(r"BERT[-\s]?(?:base|large|tiny|mini)?", re.IGNORECASE),
     "transformers", "bert-base-uncased", 0.95),
    (re.compile(r"RoBERTa[-\s]?(?:base|large)?", re.IGNORECASE),
     "transformers", "roberta-base", 0.95),
    (re.compile(r"GPT[-\s]?2?[-\s]?", re.IGNORECASE),
     "transformers", "gpt2", 0.90),
    (re.compile(r"T5[-\s]?(?:small|base|large|3b|11b)?", re.IGNORECASE),
     "transformers", "t5-base", 0.90),
    (re.compile(r"LLaMA[-\s]?[23]?[-\s]?(?:\d+)?[bB]", re.IGNORECASE),
     "transformers", None, 0.85),
    (re.compile(r"Mistral[-\s]?(?:7|8x)?[bB]", re.IGNORECASE),
     "transformers", None, 0.85),
    (re.compile(r"Qwen[-\s]?2?[-\s]?", re.IGNORECASE),
     "transformers", None, 0.85),
    (re.compile(r"CLIP", re.IGNORECASE),
     "open_clip", "ViT-B/32", 0.90),
]

# ── Component → Package mapping ────────────────────────────────────────────

COMPONENT_PACKAGES = {
    # Head architectures
    "FPN": ("mmdetection", 0.85),
    "PAFPN": ("mmdetection", 0.85),
    "BiFPN": ("timm", 0.80),
    "DETR": ("transformers", 0.85),
    "YOLO": ("ultralytics", 0.85),
    "MaskRCNN": ("detectron2", 0.85),
    "RPN": ("torchvision.models.detection", 0.85),

    # Attention variants (known)
    "CBAM": ("torch", 0.90),
    "SENet": ("torch", 0.90),
    "SE": ("torch", 0.90),
    "ECA": ("torch", 0.90),
    "FlashAttention": ("flash-attn", 0.95),
    "flash_attn": ("flash-attn", 0.95),
    "memory_efficient_attention": ("xformers", 0.90),

    # LLM-specific
    "LoRA": ("peft", 0.95),
    "QLoRA": ("peft", 0.95),
    "FSDP": ("torch.distributed.fsdp", 0.90),
    "DeepSpeed": ("deepspeed", 0.90),
    "vLLM": ("vllm", 0.85),
    "GGUF": ("llama-cpp-python", 0.85),
    "GPTQ": ("auto-gptq", 0.85),
    "AWQ": ("autoawq", 0.85),

    # Distributed/Infrastructure
    "DDP": ("torch.distributed", 0.95),
    "accelerate": ("accelerate", 0.90),
    "deepspeed": ("deepspeed", 0.90),

    # Augmentation
    "RandAugment": ("timm", 0.90),
    "CutMix": ("timm", 0.90),
    "MixUp": ("timm", 0.90),
    "AugMix": ("timm", 0.90),

    # Metrics
    "mAP": ("pycocotools", 0.90),
    "BLEU": ("sacrebleu", 0.90),
    "ROUGE": ("rouge-score", 0.90),
    "FID": ("pytorch-fid", 0.80),
}

# ── Training features ──────────────────────────────────────────────────────

TRAINING_PACKAGES = {
    "wandb": ("wandb", 0.95),
    "tensorboard": ("tensorboard", 0.95),
    "ema": ("timm.utils.model_ema", 0.85),
    "neptune": ("neptune-client", 0.90),
    "mlflow": ("mlflow", 0.90),
}

# ── Ecosystem context: what a library implies about the stack ──────────────

ECOSYSTEM_HINTS = {
    "transformers": {
        "typical_accompaniments": ["datasets", "accelerate", "tokenizers"],
        "framework": "PyTorch (primary), TF (legacy)",
        "version_note": ">=4.30.0 for most modern models",
    },
    "timm": {
        "typical_accompaniments": ["torchvision"],
        "framework": "PyTorch",
        "version_note": ">=0.9.0; API is unstable between minor versions",
    },
    "mmdetection": {
        "typical_accompaniments": ["mmcv", "mmengine"],
        "framework": "PyTorch",
        "version_note": "mmcv version must match mmdet version exactly",
    },
    "detectron2": {
        "typical_accompaniments": ["torchvision", "fvcore"],
        "framework": "PyTorch",
        "version_note": "Requires torch>=1.8; build from source on non-Linux",
    },
    "peft": {
        "typical_accompaniments": ["transformers", "bitsandbytes", "accelerate"],
        "framework": "PyTorch",
        "version_note": ">=0.7.0 for QLoRA support",
    },
    "flash-attn": {
        "typical_accompaniments": [],
        "framework": "PyTorch",
        "version_note": "Requires Ampere+ GPU; v2 needs CUDA 11.6+",
    },
    "ultralytics": {
        "typical_accompaniments": ["opencv-python", "pyyaml"],
        "framework": "PyTorch",
        "version_note": ">=8.0.0; installed via pip install ultralytics",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1: Rule-based matching
# ═══════════════════════════════════════════════════════════════════════════

def _match_backbone(arch_text: str) -> list:
    """Match architecture text against known backbone patterns."""
    results = []
    seen_patterns = set()

    for pattern, library, weight_name, confidence in BACKBONE_SOURCES:
        match = pattern.search(arch_text)
        if match and pattern.pattern not in seen_patterns:
            seen_patterns.add(pattern.pattern)
            results.append({
                "matched_text": match.group(0),
                "library": library,
                "pretrained_weight": weight_name.format(*match.groups()) if weight_name and match.groups() else weight_name,
                "confidence": confidence,
                "source": "rule",
            })

    return results


def _match_components(arch_text: str, train_text: str) -> tuple:
    """Match known components and return (matched, unmatched patterns)."""
    matched = []
    combined_text = (arch_text + " " + train_text).lower()

    for comp_name, (package, confidence) in COMPONENT_PACKAGES.items():
        if comp_name.lower() in combined_text:
            matched.append({
                "component": comp_name,
                "package": package,
                "confidence": confidence,
                "source": "rule",
            })

    return matched


def _match_training_features(train_text: str) -> list:
    """Match training features to supporting packages."""
    results = []
    train_lower = train_text.lower()

    for feature, (package, confidence) in TRAINING_PACKAGES.items():
        if feature.lower() in train_lower:
            results.append({
                "feature": feature,
                "package": package,
                "confidence": confidence,
                "source": "rule",
            })

    return results


def _deduplicate(items: list, key: str = "package") -> list:
    """Deduplicate list of dicts by a key field."""
    seen = set()
    result = []
    for item in items:
        k = item.get(key, "")
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2: LLM prompt generation for gaps
# ═══════════════════════════════════════════════════════════════════════════

def generate_llm_prompt(tier1_result: dict, parsed_architecture: dict,
                        parsed_training_recipe: dict) -> str:
    """
    Generate a structured prompt for Claude to fill dependency gaps.

    The prompt presents:
    - What we already know (rule-based matches)
    - What we're uncertain about (novel components, unmatched items)
    - Specific questions for Claude to answer
    """

    uncertain = tier1_result.get("uncertain", [])
    if not uncertain:
        return ""

    # Build a structured prompt
    lines = [
        "## Dependency Gap Analysis",
        "",
        "The following items could not be confidently matched by rule-based analysis. "
        "Please provide your best assessment based on your knowledge of the ML ecosystem.",
        "",
    ]

    # Novel components section
    novel = [u for u in uncertain if u["type"] == "novel_component"]
    if novel:
        lines.append("### Novel / Unmatched Components")
        lines.append("")
        lines.append("These components were mentioned in the paper but not found in our rule database:")
        lines.append("")
        for item in novel:
            lines.append(f"- **{item['value']}** (context: {item.get('context', 'N/A')})")
        lines.append("")
        lines.append("**Questions to answer:**")
        lines.append("1. What library/package is most likely needed for each component?")
        lines.append("2. Are any of these standard components under a different name?")
        lines.append("3. Are there version compatibility concerns for any of these?")
        lines.append("")

    # No pretrained weights section
    no_pretrained = [u for u in uncertain if u["type"] == "no_pretrained_source"]
    if no_pretrained:
        lines.append("### Backbones Without Pretrained Weight Sources")
        lines.append("")
        for item in no_pretrained:
            lines.append(f"- **{item['value']}**: matched to library `{item.get('library', 'unknown')}`")
        lines.append("")
        lines.append("**Questions to answer:**")
        lines.append("1. Where can pretrained weights for these backbones be found?")
        lines.append("2. If no pretrained weights exist, what is a reasonable initialization strategy?")
        lines.append("")

    # Ecosystem hints for matched libraries
    matched_libs = tier1_result.get("matched_libraries", [])
    if matched_libs:
        lines.append("### Ecosystem Hints for Matched Libraries")
        lines.append("")
        for lib in matched_libs:
            hints = ECOSYSTEM_HINTS.get(lib, {})
            if hints:
                lines.append(f"- **{lib}**: {hints.get('framework', '')}. "
                           f"Typical accompaniments: {', '.join(hints.get('typical_accompaniments', []))}. "
                           f"Version note: {hints.get('version_note', '')}")
        lines.append("")

    # Paper context
    lines.append("### Paper Context")
    lines.append("")
    lines.append("**Architecture summary:**")
    lines.append(f"```json")
    lines.append(json.dumps(parsed_architecture, ensure_ascii=False, indent=2))
    lines.append(f"```")
    lines.append("")
    lines.append("**Training recipe summary:**")
    lines.append(f"```json")
    lines.append(json.dumps(parsed_training_recipe, ensure_ascii=False, indent=2))
    lines.append(f"```")
    lines.append("")

    # Expected output format
    lines.append("### Expected Output Format")
    lines.append("")
    lines.append("```yaml")
    lines.append("llm_suggestions:")
    lines.append("  additional_packages:")
    lines.append("    - package: \"package_name>=version\"")
    lines.append("      reason: \"why this is needed\"")
    lines.append("      confidence: 0.0-1.0")
    lines.append("  pretrained_sources:")
    lines.append("    - backbone: \"backbone_name\"")
    lines.append("      source_library: \"library\"")
    lines.append("      model_name: \"specific_model_name\"")
    lines.append("      url: \"optional_url\"")
    lines.append("      confidence: 0.0-1.0")
    lines.append("  renamed_components:")
    lines.append("    - original_in_paper: \"name in paper\"")
    lines.append("      standard_name: \"name in library\"")
    lines.append("      library: \"library_name\"")
    lines.append("  compatibility_notes: []")
    lines.append("```")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_dependencies(parsed_architecture: dict, parsed_training_recipe: dict,
                         novel_components: list = None) -> dict:
    """
    Hybrid dependency analysis: rule-based matching + LLM gap detection.

    Args:
        parsed_architecture: From reproduction_state.yaml
        parsed_training_recipe: From reproduction_state.yaml
        novel_components: List of component names from paper that didn't match
                          any existing component in the library (optional)

    Returns:
        Dict with:
          - rule_based: confident matches from Tier 1
          - uncertain: items needing LLM reasoning (Tier 2)
          - requirements_txt: generated pip requirements
          - ecosystem_notes: library-specific guidance
    """
    arch_text = json.dumps(parsed_architecture)
    train_text = json.dumps(parsed_training_recipe)

    # ── Tier 1 matching ──
    backbone_matches = _match_backbone(arch_text.lower())
    component_matches = _match_components(arch_text, train_text)
    training_matches = _match_training_features(train_text)

    # ── Collect confident results ──
    core_packages = ["torch>=2.0.0", "torchvision>=0.15.0"]
    optional_packages = []
    pretrained_sources = []
    matched_libraries = set()

    for m in backbone_matches:
        matched_libraries.add(m["library"])
        if m["library"] == "transformers":
            if "transformers>=4.30.0" not in core_packages:
                core_packages.append("transformers>=4.30.0")
        if m.get("pretrained_weight"):
            pretrained_sources.append({
                "backbone": m["matched_text"],
                "source_library": m["library"],
                "model_name": m["pretrained_weight"],
            })

    for m in component_matches:
        pkg = m["package"]
        if pkg not in core_packages and pkg not in optional_packages and pkg != "torch":
            optional_packages.append(pkg)
        matched_libraries.add(pkg.split(".")[0])

    for m in training_matches:
        pkg = m["package"]
        if pkg not in core_packages and pkg not in optional_packages and pkg != "torch":
            optional_packages.append(pkg)

    # ── Detect uncertainty (Tier 2 gaps) ──
    uncertain = []

    # Novel components: passed in or detected as architecture fields that aren't matched
    if novel_components:
        for comp in novel_components:
            uncertain.append({
                "type": "novel_component",
                "value": comp,
                "context": "User-reported novel component from parsed architecture",
            })

    # Check for unmatched architecture fields
    arch_fields = ["backbone", "neck", "head", "loss_functions", "novel_components"]
    for field in arch_fields:
        value = parsed_architecture.get(field, "")
        if value and isinstance(value, str) and value.strip():
            # Check if this value was matched by any rule
            matched = False
            for bm in backbone_matches:
                if bm["matched_text"].lower() in value.lower():
                    matched = True
                    break
            for cm in component_matches:
                if cm["component"].lower() in value.lower():
                    matched = True
                    break
            if not matched:
                uncertain.append({
                    "type": "novel_component",
                    "value": str(value),
                    "context": f"From parsed_architecture.{field}",
                })

    # Check backbones with matched library but no pretrained weight
    for bm in backbone_matches:
        if not bm.get("pretrained_weight") and bm["library"] != "torchvision.ops":
            uncertain.append({
                "type": "no_pretrained_source",
                "value": bm["matched_text"],
                "library": bm["library"],
            })

    # ── Ecosystem notes ──
    ecosystem_notes = []
    for lib in matched_libraries:
        hints = ECOSYSTEM_HINTS.get(lib)
        if hints:
            ecosystem_notes.append({
                "library": lib,
                "hints": hints,
            })
            for acc in hints.get("typical_accompaniments", []):
                if acc not in core_packages and acc not in optional_packages:
                    optional_packages.append(acc)

    # ── Build requirements.txt ──
    all_packages = core_packages + optional_packages
    seen_pkg = set()
    deduped_all = []
    for p in all_packages:
        if p not in seen_pkg:
            seen_pkg.add(p)
            deduped_all.append(p)

    requirements_lines = [
        "# Generated by dependency_mapper.py (hybrid: rule + LLM gap analysis)",
        "# Items marked [RULE] were matched deterministically.",
        "# Items marked [LLM] need human or Claude review.",
        "",
    ]
    for p in deduped_all:
        requirements_lines.append(p)

    # ── Assemble result ──
    result = {
        "analysis_mode": "hybrid",
        "rule_based": {
            "backbone_matches": backbone_matches,
            "component_matches": component_matches,
            "training_matches": training_matches,
            "pretrained_sources": pretrained_sources,
        },
        "uncertain": uncertain,
        "needs_llm_review": len(uncertain) > 0,
        "matched_libraries": list(matched_libraries),
        "core_packages": core_packages,
        "optional_packages": optional_packages,
        "ecosystem_notes": ecosystem_notes,
        "requirements_txt": "\n".join(requirements_lines),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid dependency mapper: rule-based + LLM gap detection"
    )
    parser.add_argument("state_file", help="Path to reproduction_state.yaml")
    parser.add_argument("--output", "-o", default=None,
                        help="Write analysis to file")
    parser.add_argument("--requirements", action="store_true",
                        help="Only output requirements.txt")
    parser.add_argument("--llm-prompt", action="store_true",
                        help="Generate LLM reasoning prompt for gaps (Tier 2)")

    args = parser.parse_args()

    # Read state file
    try:
        import yaml
        with open(args.state_file, "r") as f:
            state = yaml.safe_load(f)
    except ImportError:
        with open(args.state_file, "r") as f:
            state = json.load(f)

    arch = state.get("parsed_architecture", {})
    train = state.get("parsed_training_recipe", {})
    novel = state.get("parsed_unclear_areas", [])

    result = analyze_dependencies(arch, train, novel)

    if args.requirements:
        print(result["requirements_txt"])
        return

    if args.llm_prompt:
        prompt = generate_llm_prompt(result, arch, train)
        if args.output:
            with open(args.output, "w") as f:
                f.write(prompt)
            print(f"LLM prompt written to {args.output}")
        else:
            print(prompt)
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Dependency analysis written to {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
