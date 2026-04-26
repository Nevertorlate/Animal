from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "diffsynth-studio") not in sys.path:
    sys.path.insert(0, str(ROOT / "diffsynth-studio"))

from playerone_repro import PlayerOnePipeline, PlayerOneTrainingHarness, TrainingClipDataset, build_tiny_playerone_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PlayerOne training on a local JSONL manifest.")
    parser.add_argument("--manifest", default=None, help="Path to a training manifest JSONL file. Defaults to <dataset-root>/manifest.jsonl.")
    parser.add_argument("--dataset-root", default=None, help="Dataset root used when --manifest is omitted.")
    parser.add_argument("--output-dir", default="runs/playerone_train", help="Directory for checkpoints and logs.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32", help="Pipeline dtype.")
    parser.add_argument("--tiny", action="store_true", help="Use the built-in tiny random pipeline for smoke testing.")
    parser.add_argument("--wan-model-id", default=None, help="Optional Wan model id override for real training.")
    parser.add_argument("--disable-scene-memory", action="store_true", help="Disable latent scene-memory conditioning.")
    parser.add_argument("--scene-memory-max-latent-frames", type=int, default=32, help="Maximum latent history length to retain.")
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage2", help="Training stage to enable.")
    parser.add_argument("--stage1-lora-rank", type=int, default=16, help="LoRA rank for stage-1 smoke training.")
    parser.add_argument("--stage1-lora-alpha", type=int, default=16, help="LoRA alpha for stage-1 smoke training.")
    parser.add_argument("--stage2-last-blocks", type=int, default=6, help="Number of Wan blocks to unfreeze in stage-2.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of dataset passes.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum optimization steps. Use <= 0 for no explicit cap.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset order each epoch.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping threshold. Use <= 0 to disable.")
    parser.add_argument("--log-every", type=int, default=1, help="Log interval in optimizer steps.")
    parser.add_argument("--save-every", type=int, default=0, help="Checkpoint interval in optimizer steps. Use <= 0 to disable.")
    parser.add_argument("--chunk-frames", type=int, default=9, help="Frames per autoregressive training chunk.")
    parser.add_argument("--sigma-shift", type=float, default=5.0, help="Scheduler sigma shift.")
    parser.add_argument(
        "--scene-update-mode",
        choices=("teacher_forced", "generated", "mixed"),
        default="teacher_forced",
        help="How scene state is updated between chunks during training.",
    )
    parser.add_argument("--teacher-forced-warmup-chunks", type=int, default=1, help="Warmup chunks for mixed scene update mode.")
    parser.add_argument("--rollout-negative-prompt", default="", help="Negative prompt used for generated rollouts.")
    parser.add_argument("--rollout-cfg-scale", type=float, default=1.0, help="CFG scale for generated rollouts.")
    parser.add_argument("--rollout-num-inference-steps", type=int, default=6, help="Inference steps for generated rollouts.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument("--rollout-seed", type=int, default=1000, help="Base seed for rollout generation. Use a negative value to disable.")
    return parser.parse_args()


def parse_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def resolve_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        return Path(args.manifest).resolve()
    if args.dataset_root is not None:
        return (Path(args.dataset_root) / "manifest.jsonl").resolve()
    raise ValueError("Pass either --manifest or --dataset-root.")


def build_pipeline(args: argparse.Namespace) -> PlayerOnePipeline:
    torch_dtype = parse_torch_dtype(args.dtype)
    if args.tiny and args.device == "cpu" and torch_dtype != torch.float32:
        print("tiny pipeline on CPU falls back to float32 for compatibility")
        torch_dtype = torch.float32

    if args.tiny:
        pipe = build_tiny_playerone_pipeline(device=args.device, torch_dtype=torch_dtype)
    else:
        pipe = PlayerOnePipeline.from_wan_pretrained(
            device=args.device,
            torch_dtype=torch_dtype,
            model_id_override=args.wan_model_id,
            use_scene_memory=not args.disable_scene_memory,
            scene_memory_max_latent_frames=args.scene_memory_max_latent_frames,
        )
    pipe.use_scene_memory = not args.disable_scene_memory
    pipe.scene_memory_max_latent_frames = args.scene_memory_max_latent_frames
    return pipe


def configure_training_stage(harness: PlayerOneTrainingHarness, args: argparse.Namespace) -> None:
    if args.stage == "stage1":
        harness.apply_stage1_lora(
            lora_rank=args.stage1_lora_rank,
            lora_alpha=args.stage1_lora_alpha,
        )
    else:
        harness.apply_stage2_finetune(last_blocks=args.stage2_last_blocks)


def single_item_collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(f"Expected a batch size of 1, got {len(batch)}")
    return batch[0]


def save_checkpoint(
    harness: PlayerOneTrainingHarness,
    *,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    last_loss: float,
) -> Path:
    checkpoint_path = output_dir / f"trainables-step-{step:06d}.pt"
    payload = {
        "step": step,
        "last_loss": last_loss,
        "config": vars(args),
        "state_dict": harness.export_trainable_state_dict(harness.state_dict()),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest_path(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    dataset = TrainingClipDataset(manifest_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=single_item_collate,
    )

    pipe = build_pipeline(args)
    harness = PlayerOneTrainingHarness(pipe)
    configure_training_stage(harness, args)
    harness.train()

    trainable_params = [param for param in harness.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters were enabled by the selected training stage.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    log_path = output_dir / "train_log.jsonl"
    step = 0
    last_loss = float("nan")
    print(f"loaded {len(dataset)} samples from {manifest_path}")
    print(f"trainable parameters: {sum(param.numel() for param in trainable_params):,}")

    stop_due_to_max_steps = False
    max_steps = args.max_steps if args.max_steps > 0 else None
    rollout_seed = None if args.rollout_seed < 0 else args.rollout_seed

    for epoch in range(args.epochs):
        for sample in dataloader:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = harness.compute_flow_matching_loss(
                video=sample["video"],
                prompt=sample["prompt"],
                motion=sample["motion"],
                point_maps=sample["point_maps"],
                chunk_num_frames=args.chunk_frames,
                sigma_shift=args.sigma_shift,
                scene_update_mode=args.scene_update_mode,
                teacher_forced_warmup_chunks=args.teacher_forced_warmup_chunks,
                rollout_negative_prompt=args.rollout_negative_prompt,
                rollout_cfg_scale=args.rollout_cfg_scale,
                rollout_num_inference_steps=args.rollout_num_inference_steps,
                rollout_seed=None if rollout_seed is None else rollout_seed + step,
            )
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            log_record = {
                "step": step,
                "epoch": epoch,
                "sample_id": sample["sample_id"],
                "loss": last_loss,
            }
            with log_path.open("a") as fp:
                fp.write(json.dumps(log_record, ensure_ascii=True) + "\n")

            if args.log_every > 0 and step % args.log_every == 0:
                print(f"step={step} epoch={epoch} sample={sample['sample_id']} loss={last_loss:.6f}")

            if args.save_every > 0 and step % args.save_every == 0:
                checkpoint_path = save_checkpoint(harness, output_dir=output_dir, step=step, args=args, last_loss=last_loss)
                print(f"saved checkpoint: {checkpoint_path}")

            if max_steps is not None and step >= max_steps:
                stop_due_to_max_steps = True
                break
        if stop_due_to_max_steps:
            break

    final_checkpoint = save_checkpoint(harness, output_dir=output_dir, step=step, args=args, last_loss=last_loss)
    print(f"final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
