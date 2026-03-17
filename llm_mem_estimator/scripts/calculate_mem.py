#!/usr/bin/env python3
"""
LLM Memory Estimator - CLI Interface
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import llm_mem_estimator package
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_mem_estimator import (
    ConfigLoader,
    MemoryEstimator,
    ReportGenerator,
    WeightClassifier,
    ConfigGenerator,
)


def main():
    parser = argparse.ArgumentParser(description="LLM Memory Estimator")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--config', type=str, help="Path to model YAML config file")
    input_group.add_argument('--model', type=str, help="HuggingFace model name")
    input_group.add_argument('--local', type=str, help="Path to local model weights")

    # Generation options
    parser.add_argument('--generate-config', action='store_true',
                       help="Generate YAML config from model and save it")
    parser.add_argument('--output-config', type=str,
                       help="Output path for generated config (default: configs/models/<model_name>.yaml)")

    # Estimation parameters
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size")
    parser.add_argument('--prompt-len', type=int, required=True, help="Input prompt length")
    parser.add_argument('--gen-len', type=int, default=None, help="Generated output length (required without --find-max-seq-len)")
    parser.add_argument('--kv-dtype', type=str, default="fp16", help="KV cache dtype")
    parser.add_argument('--activation-dtype', type=str, default="fp16", help="Activation dtype")

    # Parallel configuration
    parser.add_argument('--tp', type=int, default=1, help="Tensor parallel size")
    parser.add_argument('--pp', type=int, default=1, help="Pipeline parallel size")
    parser.add_argument('--dp', type=int, default=1, help="Data parallel size")
    parser.add_argument('--cp', type=int, default=1, help="Context parallel size")
    parser.add_argument('--ep', type=int, default=1, help="Expert parallel size")

    # Hardware configuration
    parser.add_argument('--chip', type=str, help="Chip name (e.g., nvidia/H100-80GB)")
    parser.add_argument('--find-max-seq-len', action='store_true',
                       help="Find maximum sequence length for given hardware")

    # System configuration
    parser.add_argument('--system-reserved', type=float, default=2.0,
                       help="System reserved memory in GB")

    # Output options
    parser.add_argument('--output', type=str, help="Output report path")

    args = parser.parse_args()

    # Validate: gen-len is required when NOT using --find-max-seq-len
    if not args.find_max_seq_len and args.gen_len is None:
        parser.error("--gen-len is required when not using --find-max-seq-len")

    # Validate: --gen-len cannot be used with --find-max-seq-len
    if args.find_max_seq_len and args.gen_len is not None:
        parser.error("--gen-len cannot be used with --find-max-seq-len")

    # Get script directory
    script_dir = Path(__file__).parent.parent

    # Load weight mapping rules
    rules_path = script_dir / "configs" / "weight_mapping_rules.yaml"
    if not rules_path.exists():
        print(f"Error: Weight mapping rules not found at {rules_path}", file=sys.stderr)
        sys.exit(1)

    rules = ConfigLoader.load_weight_mapping_rules(str(rules_path))
    classifier = WeightClassifier(rules)

    # Load or generate config
    if args.config:
        # Load from YAML config
        config = ConfigLoader.load_yaml_config(args.config)
        model_name = config.model_identity.name
    else:
        # Generate config from model
        generator = ConfigGenerator(classifier)

        if args.model:
            print(f"Generating config from HuggingFace model: {args.model}")
            config = generator.generate_config(args.model, is_local=False)
            model_name = args.model.split('/')[-1]
        else:
            print(f"Generating config from local weights: {args.local}")
            config = generator.generate_config(args.local, is_local=True)
            model_name = Path(args.local).name

        # Save generated config if requested
        if args.generate_config:
            output_config_path = args.output_config or str(script_dir / "configs" / "models" / f"{model_name}.yaml")
            Path(output_config_path).parent.mkdir(parents=True, exist_ok=True)
            ConfigLoader.save_yaml_config(config, output_config_path)
            print(f"Config saved to: {output_config_path}")

            if not args.find_max_seq_len:
                return

    # Create estimator
    estimator = MemoryEstimator(config)

    # Load chip info if specified
    chip_info = None
    available_memory_gb = None
    if args.chip:
        chips_path = script_dir / "configs" / "chips.json"
        if chips_path.exists():
            chips_config = ConfigLoader.load_chips_config(str(chips_path))
            # Search in nested vendor structure
            for vendor, chips in chips_config.items():
                if args.chip in chips:
                    chip_info = chips[args.chip].copy()
                    chip_info['name'] = f"{vendor}/{args.chip}"
                    available_memory_gb = chip_info.get('vram_gb')
                    break
            if not chip_info:
                print(f"Warning: Chip '{args.chip}' not found in chips.json", file=sys.stderr)

    # Find max sequence length if requested
    if args.find_max_seq_len:
        if not available_memory_gb:
            print("Error: --chip must be specified with --find-max-seq-len", file=sys.stderr)
            sys.exit(1)

        max_gen_len = estimator.find_max_sequence_length(
            available_memory_gb=available_memory_gb,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            kv_dtype=args.kv_dtype,
            activation_dtype=args.activation_dtype,
            tp=args.tp,
            pp=args.pp,
            cp=args.cp,
            ep=args.ep,
            system_reserved_gb=args.system_reserved
        )

        # Generate report with max sequence length
        result = estimator.estimate_memory(
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            gen_len=max_gen_len,
            kv_dtype=args.kv_dtype,
            activation_dtype=args.activation_dtype,
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            cp=args.cp,
            ep=args.ep,
            system_reserved_gb=args.system_reserved
        )

        parallel_config = {
            'tp': args.tp,
            'pp': args.pp,
            'dp': args.dp,
            'cp': args.cp,
            'ep': args.ep
        }

        report = ReportGenerator.generate_report(
            config=config,
            result=result,
            batch_size=args.batch_size,
            parallel_config=parallel_config,
            prompt_len=args.prompt_len,
            gen_len=max_gen_len,
            chip_info=chip_info
        )

        print(report)

        # Print max sequence length calculation explanation
        print("\n## How to Calculate the Maximum Generated Length")
        print("")

        # Calculate fixed memory
        weights_memory, _ = estimator.calculate_weights_memory(
            tp=args.tp, pp=args.pp, cp=args.cp, ep=args.ep
        )
        fixed_memory = weights_memory + args.system_reserved

        # KV for prompt (fixed)
        prompt_kv_memory = estimator.calculate_kv_cache_memory(
            args.batch_size, args.prompt_len, 0, args.kv_dtype, args.tp, args.cp
        )

        print("### Fixed Memory")
        print(f"- Model weights (per device, after TP/EP sharding): {weights_memory:.2f} GB")
        print(f"- System reserved: {args.system_reserved:.2f} GB")
        print(f"- KV Cache for prompt ({args.prompt_len:,}): {prompt_kv_memory:.2f} GB")
        print(f"- **Total fixed memory: {fixed_memory + prompt_kv_memory:.2f} GB**")

        available_dyn = available_memory_gb - fixed_memory - prompt_kv_memory

        print(f"\n### Available Memory for Generation")
        print(f"- Total chip VRAM: {available_memory_gb} GB")
        print(f"- Available for gen KV + Activation: {available_dyn:.2f} GB")

        # Calculate per-unit gen memory (batch_size=1, prompt_len+gen_len=1)
        # This gives the memory cost for a single token
        kv_memory_per_gen = estimator.calculate_kv_cache_memory(
            1, 0, 1, args.kv_dtype, args.tp, args.cp
        )
        act_memory_per_gen = estimator.calculate_activation_memory(
            1, 1, args.activation_dtype, args.tp, args.cp
        )
        total_per_gen = kv_memory_per_gen + act_memory_per_gen

        print(f"\n### Memory per Unit Generated Token")
        print(f"- KV Cache (incremental): {kv_memory_per_gen:.6f} GB")
        print(f"- Activation: {act_memory_per_gen:.6f} GB")
        print(f"- **Total: {total_per_gen:.6f} GB**")

        print(f"\n### Calculation")
        print(f"Max gen_len = Available memory / Memory per generated token")
        print(f"= {available_dyn:.2f} / {total_per_gen:.6f}")
        print(f"= **{max_gen_len:,}**")

        print(f"\n### Result")
        print(f"- **Maximum generated length: {max_gen_len:,}**")

        return

    # Estimate memory (use default gen_len if not specified)
    effective_gen_len = args.gen_len if args.gen_len is not None else 1024

    result = estimator.estimate_memory(
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        gen_len=effective_gen_len,
        kv_dtype=args.kv_dtype,
        activation_dtype=args.activation_dtype,
        tp=args.tp,
        pp=args.pp,
        dp=args.dp,
        cp=args.cp,
        ep=args.ep,
        system_reserved_gb=args.system_reserved
    )

    # Generate report
    parallel_config = {
        'tp': args.tp,
        'pp': args.pp,
        'dp': args.dp,
        'cp': args.cp,
        'ep': args.ep
    }

    report = ReportGenerator.generate_report(
        config=config,
        result=result,
        batch_size=args.batch_size,
        parallel_config=parallel_config,
        prompt_len=args.prompt_len,
        gen_len=effective_gen_len,
        chip_info=chip_info
    )

    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
