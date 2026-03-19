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
    parser.add_argument('--prompt-len', type=int, default=None, help="Input prompt length (required with --find-max-seq-len)")
    parser.add_argument('--gen-len', type=int, default=None, help="Generated output length (required without --find-max-seq-len)")
    parser.add_argument('--kv-dtype', type=str, default="fp16", help="KV cache dtype")
    parser.add_argument('--activation-dtype', type=str, default="fp16", help="Activation dtype")
    parser.add_argument('--activation-peak', type=float, default=None,
                       help="Fixed activation peak value in GB (overrides formula calculation)")

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

    # Skip validation for --generate-config (only generates config, no memory calculation needed)
    if args.generate_config:
        # Generate config and exit
        script_dir = Path(__file__).parent.parent
        rules_path = script_dir / "configs" / "weight_mapping_rules.yaml"
        if not rules_path.exists():
            print(f"Error: Weight mapping rules not found at {rules_path}", file=sys.stderr)
            sys.exit(1)

        rules = ConfigLoader.load_weight_mapping_rules(str(rules_path))
        classifier = WeightClassifier(rules)
        generator = ConfigGenerator(classifier)

        if args.model:
            print(f"Generating config from HuggingFace model: {args.model}")
            config = generator.generate_config(args.model, is_local=False)
            model_name = args.model.split('/')[-1]
        elif args.local:
            print(f"Generating config from local weights: {args.local}")
            config = generator.generate_config(args.local, is_local=True)
            model_name = Path(args.local).name
        else:
            print("Error: --model or --local required with --generate-config", file=sys.stderr)
            sys.exit(1)

        output_config_path = args.output_config or str(script_dir / "configs" / "models" / f"{model_name}.yaml")
        Path(output_config_path).parent.mkdir(parents=True, exist_ok=True)
        ConfigLoader.save_yaml_config(config, output_config_path)
        print(f"Config saved to: {output_config_path}")
        return

    # Validation logic for different scenarios:
    # Scene 1: --find-max-seq-len=✗, --gen-len=✓, --prompt-len=✓ → normal estimation
    # Scene 2: --find-max-seq-len=✗, --gen-len=✓, --prompt-len=✗ → error
    # Scene 3: --find-max-seq-len=✗, --gen-len=✗, --prompt-len=✓ → error
    # Scene 4: --find-max-seq-len=✗, --gen-len=✗, --prompt-len=✗ → error
    # Scene 5: --find-max-seq-len=✓, --gen-len=✗, --prompt-len=✗ → search max gen_len, default prompt_len=4096
    # Scene 6: --find-max-seq-len=✓, --gen-len=✗, --prompt-len=✓ → search max gen_len with user prompt_len
    # Scene 7: --find-max-seq-len=✓, --gen-len=✓, --prompt-len=✗ → search max prompt_len with user gen_len
    # Scene 8: --find-max-seq-len=✓, --gen-len=✓, --prompt-len=✓ → warn, treat as Scene 1

    if not args.find_max_seq_len:
        # Scenes 1-4: --find-max-seq-len is False
        if args.gen_len is None or args.prompt_len is None:
            parser.error("--prompt-len and --gen-len are required when not using --find-max-seq-len")

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

    # Create estimator
    estimator = MemoryEstimator(config)

    # Load chip info if specified
    chip_info = None
    available_memory_gb = None
    if args.chip:
        chips_path = script_dir / "configs" / "chips.json"
        if chips_path.exists():
            chips_config = ConfigLoader.load_chips_config(str(chips_path))
            # Parse chip name - support both "Vendor/ChipName" and "ChipName" formats
            chip_name = args.chip
            if '/' in args.chip:
                parts = args.chip.split('/', 1)
                vendor = parts[0]
                chip_name = parts[1]
                # Look up in specific vendor
                if vendor in chips_config and chip_name in chips_config[vendor]:
                    chip_info = chips_config[vendor][chip_name].copy()
                    chip_info['name'] = f"{vendor}/{chip_name}"
                    available_memory_gb = chip_info.get('vram_gb')
            else:
                # Short name format - search in all vendors
                for vendor, chips in chips_config.items():
                    if chip_name in chips:
                        chip_info = chips[chip_name].copy()
                        chip_info['name'] = f"{vendor}/{chip_name}"
                        available_memory_gb = chip_info.get('vram_gb')
                        break
            if not chip_info:
                # Build supported chips list grouped by vendor
                vendor_groups = []
                for vendor, vendor_chips in chips_config.items():
                    chip_names = list(vendor_chips.keys())
                    vendor_groups.append(f"  {vendor}: {', '.join(f'{c}' for c in chip_names)}")
                supported_chips = '\n'.join(vendor_groups)
                print(f"Error: Chip '{args.chip}' not found.", file=sys.stderr)
                print(f"Supported chips:\n{supported_chips}\n(Use 'Vendor/ChipName' format, e.g., 'nvidia/H100-80GB' or short name like 'H100-80GB')", file=sys.stderr)
                sys.exit(1)

    # Find max sequence length if requested
    if args.find_max_seq_len:
        if not available_memory_gb:
            # Load chips config to show supported chips
            chips_path = script_dir / "configs" / "chips.json"
            if chips_path.exists():
                chips_config = ConfigLoader.load_chips_config(str(chips_path))
                vendor_groups = []
                for vendor, vendor_chips in chips_config.items():
                    chip_names = list(vendor_chips.keys())
                    vendor_groups.append(f"  {vendor}: {', '.join(f'{c}' for c in chip_names)}")
                supported_chips = '\n'.join(vendor_groups)
                print("Error: --chip must be specified with --find-max-seq-len", file=sys.stderr)
                print(f"Supported chips:\n{supported_chips}", file=sys.stderr)
            else:
                print("Error: --chip must be specified with --find-max-seq-len", file=sys.stderr)
            sys.exit(1)

        # Get system_reserved_gb and gpu_util from computation_rules
        computation_rules = config.computation_rules
        system_reserved_gb = computation_rules.get('system_reserved_gb', 2.0)
        gpu_util = computation_rules.get('gpu_util', 1.0)

        # Validate gpu_util
        if not isinstance(gpu_util, (int, float)) or gpu_util <= 0 or gpu_util > 1.0:
            print(f"Error: gpu_util must be > 0 and <= 1.0, got {gpu_util}", file=sys.stderr)
            sys.exit(1)

        # Calculate actual available memory
        actual_available_memory_gb = available_memory_gb * gpu_util

        # Scene 8: --find-max-seq-len with both --gen-len and --prompt-len
        # → search max batch_size if batch_size=1, otherwise normal estimation
        if args.gen_len is not None and args.prompt_len is not None:
            effective_prompt_len = args.prompt_len
            effective_gen_len = args.gen_len
            effective_batch_size = args.batch_size

            if args.batch_size != 1:
                # batch_size != 1: do normal estimation and show fit/exceeds
                print("Note: Both --prompt-len and --gen-len specified with --find-max-seq-len, "
                      "treating as normal estimation", file=sys.stderr)
                result = estimator.estimate_memory(
                    batch_size=effective_batch_size,
                    prompt_len=effective_prompt_len,
                    gen_len=effective_gen_len,
                    kv_dtype=args.kv_dtype,
                    activation_dtype=args.activation_dtype,
                    tp=args.tp,
                    pp=args.pp,
                    dp=args.dp,
                    cp=args.cp,
                    ep=args.ep,
                    system_reserved_gb=system_reserved_gb,
                    use_decode_factor=False,
                    activation_peak_gb=args.activation_peak
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
                    batch_size=effective_batch_size,
                    parallel_config=parallel_config,
                    prompt_len=effective_prompt_len,
                    gen_len=effective_gen_len,
                    chip_info=chip_info
                )

                print(report)

                # Add memory fit conclusion if chip_info is available
                if chip_info and available_memory_gb:
                    total_memory = result.weights_memory_gb + result.kv_cache_memory_gb + result.activation_memory_gb + system_reserved_gb
                    available_with_util = available_memory_gb * gpu_util
                    remaining = available_with_util - total_memory
                    fit_status = "✅ Fits" if remaining >= 0 else "❌ Exceeds"
                    print(f"\n## Result")
                    print(f"")
                    print(f"- VRAM * gpu_util - Total = {available_memory_gb} * {gpu_util:.0%} - {total_memory:.2f} = {available_with_util:.2f} - {total_memory:.2f} = {remaining:.2f} GB")
                    print(f"- Status: {fit_status}")
                return

            # batch_size == 1: search for max batch_size
            print("Note: --batch-size=1 with --find-max-seq-len, --prompt-len and --gen-len, "
                  "searching for maximum batch_size...", file=sys.stderr)
            max_batch_size = estimator.find_max_batch_size(
                available_memory_gb=actual_available_memory_gb,
                prompt_len=effective_prompt_len,
                gen_len=effective_gen_len,
                kv_dtype=args.kv_dtype,
                activation_dtype=args.activation_dtype,
                tp=args.tp,
                pp=args.pp,
                cp=args.cp,
                ep=args.ep,
                system_reserved_gb=system_reserved_gb,
                activation_peak_gb=args.activation_peak
            )

            # Generate report with max batch size
            result = estimator.estimate_memory(
                batch_size=max_batch_size,
                prompt_len=effective_prompt_len,
                gen_len=effective_gen_len,
                kv_dtype=args.kv_dtype,
                activation_dtype=args.activation_dtype,
                tp=args.tp,
                pp=args.pp,
                dp=args.dp,
                cp=args.cp,
                ep=args.ep,
                system_reserved_gb=system_reserved_gb,
                use_decode_factor=False,
                activation_peak_gb=args.activation_peak
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
                batch_size=max_batch_size,
                parallel_config=parallel_config,
                prompt_len=effective_prompt_len,
                gen_len=effective_gen_len,
                chip_info=chip_info
            )

            print(report)

            # Add memory fit conclusion
            if chip_info and available_memory_gb:
                total_memory = result.weights_memory_gb + result.kv_cache_memory_gb + result.activation_memory_gb + system_reserved_gb
                available_with_util = available_memory_gb * gpu_util
                remaining = available_with_util - total_memory
                fit_status = "✅ Fits" if remaining >= 0 else "❌ Exceeds"
                print(f"\n## Result")
                print(f"- **Maximum batch size: {max_batch_size:,}**")
                print(f"- VRAM * gpu_util - Total = {available_memory_gb} * {gpu_util:.0%} - {total_memory:.2f} = {available_with_util:.2f} - {total_memory:.2f} = {remaining:.2f} GB")
                print(f"- Status: {fit_status}")
            return

        # Scene 7: --find-max-seq-len with --gen-len but without --prompt-len
        # → search max prompt_len with user-specified gen_len
        elif args.gen_len is not None and args.prompt_len is None:
            # Search max prompt_len
            effective_gen_len = args.gen_len
            max_prompt_len = estimator.find_max_prompt_len(
                available_memory_gb=actual_available_memory_gb,
                batch_size=args.batch_size,
                gen_len=effective_gen_len,
                kv_dtype=args.kv_dtype,
                activation_dtype=args.activation_dtype,
                tp=args.tp,
                pp=args.pp,
                cp=args.cp,
                ep=args.ep,
                system_reserved_gb=system_reserved_gb,
                activation_peak_gb=args.activation_peak
            )

            # Generate report with max prompt length (Prefill scenario)
            result = estimator.estimate_memory(
                batch_size=args.batch_size,
                prompt_len=max_prompt_len,
                gen_len=effective_gen_len,
                kv_dtype=args.kv_dtype,
                activation_dtype=args.activation_dtype,
                tp=args.tp,
                pp=args.pp,
                dp=args.dp,
                cp=args.cp,
                ep=args.ep,
                system_reserved_gb=system_reserved_gb,
                use_decode_factor=False,
                activation_peak_gb=args.activation_peak
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
                prompt_len=max_prompt_len,
                gen_len=effective_gen_len,
                chip_info=chip_info
            )

            print(report)

            # Print max prompt_len calculation explanation
            print("\n## How to Calculate the Maximum Prompt Length")
            print("")

            # Calculate fixed memory
            weights_memory, _ = estimator.calculate_weights_memory(
                tp=args.tp, pp=args.pp, cp=args.cp, ep=args.ep
            )
            fixed_memory = weights_memory + system_reserved_gb

            # Activation: use user-specified peak or calculate with formula
            if args.activation_peak is not None:
                act_memory = args.activation_peak
                act_display = f"{act_memory:.2f} (user specified, fixed)"
                # When activation is fixed, it doesn't grow with seq_len
                max_act_memory = act_memory
            else:
                # Calculate activation at max_prompt_len + gen_len
                max_act_memory = estimator.calculate_activation_memory(
                    args.batch_size, max_prompt_len + effective_gen_len, args.activation_dtype, args.tp, args.cp,
                    use_decode_factor=False
                )
                act_memory = max_act_memory
                act_display = f"{act_memory:.2f}"

            # Calculate memory breakdown for the max prompt_len
            max_kv_memory = estimator.calculate_kv_cache_memory(
                args.batch_size, max_prompt_len, effective_gen_len, args.kv_dtype, args.tp, args.cp
            )

            print("### Available Memory for Prompt")
            print(f"- Total chip VRAM: {available_memory_gb} GB")
            print(f"- GPU utilization: {gpu_util:.0%}")
            print(f"- Available memory: {actual_available_memory_gb:.2f} GB")

            print("\n### Memory Breakdown")
            print(f"- Model weights: {weights_memory:.2f} GB")
            print(f"- System reserved (from config): {system_reserved_gb:.2f} GB")
            print(f"- KV Cache (prompt_len={max_prompt_len:,} + gen_len={effective_gen_len:,}): {max_kv_memory:.2f} GB")
            print(f"- Activation (total_seq_len={max_prompt_len + effective_gen_len:,}): {act_display} GB")
            print(f"- **Total: {weights_memory + system_reserved_gb + max_kv_memory + act_memory:.2f} GB**")

            print(f"\n### Memory per Unit Prompt Token (Calculation Steps)")
            # Calculate incremental cost for adding 1 more prompt token
            kv_increment = estimator.calculate_kv_cache_memory(
                args.batch_size, max_prompt_len + 1, effective_gen_len, args.kv_dtype, args.tp, args.cp
            ) - max_kv_memory
            act_increment = estimator.calculate_activation_memory(
                args.batch_size, max_prompt_len + 1 + effective_gen_len, args.activation_dtype, args.tp, args.cp,
                use_decode_factor=False
            ) - max_act_memory
            total_increment = kv_increment + act_increment

            print("```")
            if args.activation_peak is not None:
                print(f"Prefill stage: Activation is FIXED (user specified), only KV Cache grows")
            else:
                print(f"Prefill stage: total_seq_len = prompt_len + gen_len, factor = 1.25")
            print(f"")
            print(f"KV Cache (incremental):")
            print(f"  = batch_size * seq_len * kv_dim * num_layers * dtype / (tp * cp)")
            print(f"  = {args.batch_size} * 1 * kv_dim * {config.architecture_config.num_layers} * 2 / ({args.tp} * {args.cp})")
            print(f"  = {kv_increment:.6f} GB")
            print(f"")
            if args.activation_peak is not None:
                print(f"Activation (fixed, not incremental):")
                print(f"  = {args.activation_peak:.2f} GB (user specified)")
                act_increment = 0
                total_increment = kv_increment
            else:
                print(f"Activation (incremental):")
                print(f"  = batch_size * seq_len * hidden_size * num_experts * factor * dtype / cp")
                print(f"  = {args.batch_size} * 1 * 2880 * 4 * 1.25 * 2 / {args.cp}")
                print(f"  = {act_increment:.6f} GB")
                total_increment = kv_increment + act_increment
            print(f"")
            print(f"Total (per token) = {total_increment:.6f} GB")
            print("```")

            # Calculate available memory for prompt
            available_for_prompt = actual_available_memory_gb - weights_memory - system_reserved_gb - act_memory

            print(f"\n### Calculation")
            print(f"Max prompt_len = Available memory / Total per token")
            print(f"= {available_for_prompt:.2f} GB / {total_increment:.6f} GB")
            print(f"= **{max_prompt_len:,}**")

            print(f"\n### Result")
            print(f"- **Maximum prompt length: {max_prompt_len:,}**")

            return

        # Scene 5/6: --find-max-seq-len without --gen-len
        # → search max gen_len (use default prompt_len=4096 if not specified)
        effective_prompt_len = args.prompt_len if args.prompt_len is not None else 4096

        max_gen_len = estimator.find_max_sequence_length(
            available_memory_gb=actual_available_memory_gb,
            batch_size=args.batch_size,
            prompt_len=effective_prompt_len,
            kv_dtype=args.kv_dtype,
            activation_dtype=args.activation_dtype,
            tp=args.tp,
            pp=args.pp,
            cp=args.cp,
            ep=args.ep,
            system_reserved_gb=system_reserved_gb,
            use_decode_factor=True,
            activation_peak_gb=args.activation_peak
        )

        # Generate report with max sequence length (Decode scenario)
        result = estimator.estimate_memory(
            batch_size=args.batch_size,
            prompt_len=effective_prompt_len,
            gen_len=max_gen_len,
            kv_dtype=args.kv_dtype,
            activation_dtype=args.activation_dtype,
            tp=args.tp,
            pp=args.pp,
            dp=args.dp,
            cp=args.cp,
            ep=args.ep,
            system_reserved_gb=system_reserved_gb,
            use_decode_factor=True,
            activation_peak_gb=args.activation_peak
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
            prompt_len=effective_prompt_len,
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
        fixed_memory = weights_memory + system_reserved_gb

        # KV for prompt (fixed)
        prompt_kv_memory = estimator.calculate_kv_cache_memory(
            args.batch_size, effective_prompt_len, 0, args.kv_dtype, args.tp, args.cp
        )

        # Activation: use user-specified peak or calculate with decode factor
        if args.activation_peak is not None:
            act_decode = args.activation_peak
            act_str = f"{act_decode:.2f} (user specified)"
        else:
            act_decode = estimator.calculate_activation_memory(
                args.batch_size, 1, args.activation_dtype, args.tp, args.cp,
                use_decode_factor=True
            )
            act_str = f"{act_decode:.6f}" if act_decode < 0.01 else f"{act_decode:.2f}"

        print("### Fixed Memory")
        print(f"- Model weights (per device, after TP/EP sharding): {weights_memory:.2f} GB")
        print(f"- System reserved (from config): {system_reserved_gb:.2f} GB")
        print(f"- KV Cache for prompt ({effective_prompt_len:,}): {prompt_kv_memory:.2f} GB")
        print(f"- Activation (fixed, seq_len=1): {act_str} GB")
        print(f"- **Total fixed memory: {weights_memory + system_reserved_gb + prompt_kv_memory + act_decode:.2f} GB**")

        available_dyn = actual_available_memory_gb - weights_memory - system_reserved_gb - prompt_kv_memory - act_decode

        print(f"\n### Available Memory for Generation")
        print(f"- Total chip VRAM: {available_memory_gb} GB")
        print(f"- GPU utilization: {gpu_util:.0%}")
        print(f"- Available memory: {actual_available_memory_gb:.2f} GB")
        print(f"- Available for gen KV (Activation is fixed): {available_dyn:.2f} GB")

        # Calculate per-unit gen memory: only KV Cache increments
        # In Decode stage, Activation is fixed (seq_len = 1), only KV Cache grows with gen_len
        kv_memory_per_gen = estimator.calculate_kv_cache_memory(
            args.batch_size, 0, 1, args.kv_dtype, args.tp, args.cp
        )

        print(f"\n### Memory per Unit Generated Token (Calculation Steps)")
        print("```")
        print(f"Decode stage: Activation is FIXED (seq_len=1), only KV Cache grows")
        print(f"")
        print(f"KV Cache (incremental):")
        print(f"  = batch_size * seq_len * kv_dim * num_layers * dtype / (tp * cp)")
        print(f"  = {args.batch_size} * 1 * kv_dim * {config.architecture_config.num_layers} * 2 / ({args.tp} * {args.cp})")
        print(f"  = {kv_memory_per_gen:.6f} GB")
        print(f"")
        print(f"Activation (fixed, not incremental):")
        if args.activation_peak is not None:
            print(f"  = {act_decode:.2f} GB (user specified)")
        else:
            print(f"  = batch_size * seq_len * hidden_size * num_experts * factor * dtype / cp")
            print(f"  = {args.batch_size} * 1 * 2880 * 4 * 12.5 * 2 / {args.cp}")
            print(f"  = {act_decode:.6f} GB")
        print("```")

        print(f"\n### Calculation")
        print(f"Max gen_len = Available memory / KV Cache per generated token")
        print(f"= {available_dyn:.2f} GB / {kv_memory_per_gen:.6f} GB")
        print(f"= **{max_gen_len:,}**")

        print(f"\n### Result")
        print(f"- **Maximum generated length: {max_gen_len:,}**")

        return

    # Estimate memory (use default values if not specified)
    effective_prompt_len = args.prompt_len if args.prompt_len is not None else 4096
    effective_gen_len = args.gen_len if args.gen_len is not None else 1024

    # Get system_reserved_gb and gpu_util from computation_rules
    computation_rules = config.computation_rules
    system_reserved_gb = computation_rules.get('system_reserved_gb', 2.0)
    gpu_util = computation_rules.get('gpu_util', 1.0)

    # Validate gpu_util
    if not isinstance(gpu_util, (int, float)) or gpu_util <= 0 or gpu_util > 1.0:
        print(f"Error: gpu_util must be > 0 and <= 1.0, got {gpu_util}", file=sys.stderr)
        sys.exit(1)

    result = estimator.estimate_memory(
        batch_size=args.batch_size,
        prompt_len=effective_prompt_len,
        gen_len=effective_gen_len,
        kv_dtype=args.kv_dtype,
        activation_dtype=args.activation_dtype,
        tp=args.tp,
        pp=args.pp,
        dp=args.dp,
        cp=args.cp,
        ep=args.ep,
        system_reserved_gb=system_reserved_gb,
        use_decode_factor=False,
        activation_peak_gb=args.activation_peak
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
        prompt_len=effective_prompt_len,
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

    # Add memory fit conclusion if chip_info is available
    if chip_info and available_memory_gb:
        total_memory = result.weights_memory_gb + result.kv_cache_memory_gb + result.activation_memory_gb + system_reserved_gb
        available_with_util = available_memory_gb * gpu_util
        remaining = available_with_util - total_memory
        fit_status = "✅ Fits" if remaining >= 0 else "❌ Exceeds"
        print(f"\n## Result")
        print(f"")
        print(f"- VRAM * gpu_util - Total = {available_memory_gb} * {gpu_util:.0%} - {total_memory:.2f} = {available_with_util:.2f} - {total_memory:.2f} = {remaining:.2f} GB")
        print(f"- Status: {fit_status}")


if __name__ == "__main__":
    main()
