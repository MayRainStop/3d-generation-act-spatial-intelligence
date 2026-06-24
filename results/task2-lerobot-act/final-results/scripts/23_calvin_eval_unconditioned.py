#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_EVAL_PATH = ROOT / "scripts" / "15_calvin_eval_taskcond.py"
spec = importlib.util.spec_from_file_location("hw3_base_calvin_eval", BASE_EVAL_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


class UnconditionedACTCalvinModel:
    def __init__(self, checkpoint: Path, device: str):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint, config_filename="policy_preprocessor.json", local_files_only=True
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint, config_filename="policy_postprocessor.json", local_files_only=True
        )
        self.current_mapping: dict | None = None

    def reset(self, subtask: str, goal: str) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.current_mapping = {
            "subtask": subtask,
            "goal": goal,
            "condition_type": "unconditioned",
            "condition_dim": 0,
        }

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)

    def _make_observation(self, obs: dict) -> dict[str, torch.Tensor]:
        return {
            "observation.images.top": self._image_to_tensor(obs["rgb_obs"]["rgb_static"]),
            "observation.images.wrist": self._image_to_tensor(obs["rgb_obs"]["rgb_gripper"]),
            "observation.state": torch.from_numpy(np.asarray(obs["robot_obs"], dtype=np.float32)),
        }

    def step(self, obs: dict) -> np.ndarray:
        batch = self.preprocessor.process_observation(self._make_observation(obs))
        with torch.no_grad():
            action = self.policy.select_action(batch)
            action = self.postprocessor.process_action(action)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float64)
        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
        action_np[:6] = np.clip(action_np[:6], -1.0, 1.0)
        action_np[6] = 1.0 if action_np[6] >= 0 else -1.0
        return action_np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", default="calvin_unconditioned_eval")
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--sequence-workers", type=int, default=1)
    parser.add_argument("--ep-len", type=int, default=360)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-log-dir", default=str(ROOT / "results" / "calvin_eval_extensions"))
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    base.patch_numpy_legacy_aliases()
    base.add_calvin_paths()
    dataset_path = Path(args.dataset_path).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    eval_log_dir = Path(args.eval_log_dir).resolve()
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    status_file = Path(args.status_file or (ROOT / "results" / f"{args.run_id}.status.json"))
    detail_file = eval_log_dir / f"{args.run_id}_details.json"
    summary_file = eval_log_dir / f"{args.run_id}_summary.json"
    start_time = time.time()
    status = {
        "run_id": args.run_id,
        "status": "initializing",
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint),
        "condition_type": "unconditioned",
        "num_sequences": args.num_sequences,
        "ep_len": args.ep_len,
        "completed_sequences": 0,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    base.write_status(status_file, status)
    env = base.make_env(dataset_path)
    task_oracle = base.make_task_oracle()
    val_annotations = base.load_validation_annotations()
    model = UnconditionedACTCalvinModel(checkpoint, args.device)
    eval_sequences = base.load_official_get_sequences()(args.num_sequences, num_workers=args.sequence_workers)
    results: list[int] = []
    details = []
    status["status"] = "running"
    status["total_sequences"] = len(eval_sequences)
    base.write_status(status_file, status)
    try:
        for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences, start=1):
            seq_start = time.time()
            result, subtask_logs = base.evaluate_sequence(
                env, model, task_oracle, initial_state, eval_sequence, val_annotations, args.ep_len, args.debug
            )
            results.append(int(result))
            details.append(
                {
                    "sequence_index": seq_idx,
                    "initial_state": dict(initial_state),
                    "eval_sequence": list(eval_sequence),
                    "successful_subtasks": int(result),
                    "duration_sec": time.time() - seq_start,
                    "subtasks": subtask_logs,
                }
            )
            success_rates = base.count_success(results)
            status.update(
                {
                    "completed_sequences": seq_idx,
                    "latest_successful_subtasks": int(result),
                    "avg_successful_sequence_len": float(np.mean(results)),
                    "success_rates": {str(i + 1): success_rates[i] for i in range(5)},
                    "elapsed_sec": time.time() - start_time,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "latest_mapping": subtask_logs[-1]["mapping"] if subtask_logs else None,
                }
            )
            if seq_idx == 1 or seq_idx % 10 == 0 or seq_idx == len(eval_sequences):
                base.write_status(status_file, status)
                detail_file.write_text(json.dumps(details, indent=2), encoding="utf-8")
                print(
                    f"[{args.run_id}] {seq_idx}/{len(eval_sequences)} "
                    f"avg_len={status['avg_successful_sequence_len']:.3f} "
                    f"sr1={success_rates[0] * 100:.1f}%",
                    flush=True,
                )
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": repr(exc),
                "completed_sequences": len(results),
                "elapsed_sec": time.time() - start_time,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        base.write_status(status_file, status)
        detail_file.write_text(json.dumps(details, indent=2), encoding="utf-8")
        raise
    summary = {
        "run_id": args.run_id,
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint),
        "condition_type": "unconditioned",
        "num_sequences": len(eval_sequences),
        "ep_len": args.ep_len,
        "avg_successful_sequence_len": float(np.mean(results)) if results else 0.0,
        "success_rates": {str(i + 1): base.count_success(results)[i] for i in range(5)},
        "results": results,
        "elapsed_sec": time.time() - start_time,
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detail_file.write_text(json.dumps(details, indent=2), encoding="utf-8")
    status.update(
        {
            "status": "finished",
            "exit_code": 0,
            "summary_file": str(summary_file),
            "detail_file": str(detail_file),
            "elapsed_sec": time.time() - start_time,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    base.write_status(status_file, status)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
