from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2.checkpointer.dcp import (
    DefaultLoadPlanner,
    DistributedCheckpointer,
    ModelWrapper,
    dcp_load_state_dict,
)


def export_ema_bf16(
    experiment_name: str,
    config_file: str,
    ckpt_dir: str,
    output_dir: str,
    prefix: str,
    export_action_head: bool,
    save_dtype: str,
) -> tuple[str, str | None]:
    ckpt_dir = str(Path(ckpt_dir).resolve())
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if save_dtype not in dtype_map:
        raise ValueError(f"Unsupported save_dtype: {save_dtype}. Use bf16/fp32.")
    target_dtype = dtype_map[save_dtype]

    if not prefix:
        prefix = Path(ckpt_dir).name  # e.g. iter_000003000

    # Build model first, then load weights.
    # This avoids easy_io local-path limitations for DCP directories.
    model, config = load_model_from_checkpoint(
        experiment_name=experiment_name,
        s3_checkpoint_dir=ckpt_dir,
        config_file=config_file,
        enable_fsdp=False,
        load_ema_to_reg=False,
        skip_load_model=True,
    )
    ckpt_path = Path(ckpt_dir)
    dcp_model_dir = ckpt_path
    if ckpt_path.is_dir() and (ckpt_path / "model").exists():
        dcp_model_dir = ckpt_path / "model"
    if dcp_model_dir.is_dir():
        distcp_files = list(dcp_model_dir.glob("*.distcp"))
        if len(distcp_files) == 0:
            raise FileNotFoundError(f"No *.distcp files found under {dcp_model_dir}")
        # DCP load path
        checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
        wrapper = ModelWrapper(model, load_ema_to_reg=False)
        state_dict = wrapper.state_dict()
        storage_reader = checkpointer.get_storage_reader(str(dcp_model_dir))
        load_planner = DefaultLoadPlanner(allow_partial_load=True)
        dcp_load_state_dict(state_dict, storage_reader, load_planner)
        wrapper.load_state_dict(state_dict)
    else:
        # Fallback: .pt path
        model, _ = load_model_from_checkpoint(
            experiment_name=experiment_name,
            s3_checkpoint_dir=ckpt_dir,
            config_file=config_file,
            enable_fsdp=False,
            load_ema_to_reg=False,
        )
    model.eval()

    sd = model.state_dict()
    ema_as_net = {}
    for k, v in sd.items():
        if k.startswith("net_ema."):
            ema_as_net["net." + k[len("net_ema.") :]] = v.detach().cpu().to(target_dtype)

    if len(ema_as_net) == 0:
        raise RuntimeError("No `net_ema.*` keys found in loaded checkpoint.")

    out_pt = out_dir / f"{prefix}_ema_bf16.pt"
    payload = {
        "model": ema_as_net,
        "meta": {
            "source_ckpt_dir": ckpt_dir,
            "export_type": "ema_as_net",
            "dtype": save_dtype,
        },
    }
    torch.save(payload, str(out_pt))

    action_head_out = None
    if export_action_head and hasattr(model, "action_head") and model.action_head is not None:
        action_head_out = out_dir / f"{prefix}_action_head.pt"
        torch.save({"action_head": model.action_head.state_dict()}, str(action_head_out))

    return str(out_pt), (str(action_head_out) if action_head_out else None)


def validate_pt_loadable(experiment_name: str, config_file: str, pt_path: str) -> None:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(pt_path)
    # Reuse framework loader to ensure this .pt can be consumed by cosmos runtime.
    model, _ = load_model_from_checkpoint(
        experiment_name=experiment_name,
        s3_checkpoint_dir=pt_path,
        config_file=config_file,
        enable_fsdp=False,
        load_ema_to_reg=False,
    )
    model.eval()
    print(f"[OK] validated load: {pt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export DCP checkpoint -> single ema_bf16.pt")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path like .../checkpoints/iter_000003000 or .../iter_000003000/model")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="", help="Output prefix, e.g. timestep3000")
    parser.add_argument("--export_action_head", action="store_true")
    parser.add_argument("--save_dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--validate_after_export", action="store_true")
    parser.add_argument("--validate_only_pt", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate_only_pt:
        validate_pt_loadable(args.experiment_name, args.config_file, args.validate_only_pt)
        return

    out_pt, action_head_pt = export_ema_bf16(
        experiment_name=args.experiment_name,
        config_file=args.config_file,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        export_action_head=args.export_action_head,
        save_dtype=args.save_dtype,
    )

    print(f"[SAVED] ema pt: {out_pt}")
    if action_head_pt:
        print(f"[SAVED] action head pt: {action_head_pt}")

    if args.validate_after_export:
        validate_pt_loadable(args.experiment_name, args.config_file, out_pt)


if __name__ == "__main__":
    main()
