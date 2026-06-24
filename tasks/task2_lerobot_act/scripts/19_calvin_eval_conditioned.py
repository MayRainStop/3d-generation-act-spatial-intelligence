#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_EVAL_PATH = ROOT / "scripts" / "15_calvin_eval_taskcond.py"
spec = importlib.util.spec_from_file_location("hw3_base_calvin_eval", BASE_EVAL_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().replace("_", " ").replace(":", " ").split())


def hash_embed(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in normalize_text(text).split():
        digest = zlib.crc32(token.encode("utf-8"))
        idx = digest % dim
        sign = 1.0 if ((digest >> 8) & 1) else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


class GenericConditionedACTCalvinModel:
    def __init__(
        self,
        checkpoint: Path,
        task_meta: Path,
        device: str,
        condition_type: str,
        sbert_model: str,
        hash_dim: int,
    ):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        self.device = torch.device(device)
        self.condition_type = condition_type
        self.sbert_model_name = sbert_model
        self.hash_dim = hash_dim
        self.resolver = base.TaskIndexResolver(task_meta)
        self.sbert_model = None
        if condition_type == "sbert":
            from sentence_transformers import SentenceTransformer

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HOME")
            self.sbert_model = SentenceTransformer(sbert_model, cache_folder=cache_folder)
        elif condition_type not in {"onehot", "zero_onehot", "wrong_onehot", "sbert", "hash"}:
            raise ValueError(f"Unsupported condition_type={condition_type!r}")
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
        self.current_condition: np.ndarray | None = None

    def reset(self, subtask: str, goal: str) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        mapping = self.resolver.resolve(subtask, goal)
        if self.condition_type == "onehot":
            cond = np.zeros(self.resolver.task_dim, dtype=np.float32)
            cond[int(mapping["task_index"])] = 1.0
        elif self.condition_type == "zero_onehot":
            cond = np.zeros(self.resolver.task_dim, dtype=np.float32)
        elif self.condition_type == "wrong_onehot":
            cond = np.zeros(self.resolver.task_dim, dtype=np.float32)
            cond[(int(mapping["task_index"]) + 1) % self.resolver.task_dim] = 1.0
        elif self.condition_type == "sbert":
            assert self.sbert_model is not None
            cond = self.sbert_model.encode(
                [f"{subtask}: {goal}"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].astype(np.float32)
        else:
            cond = hash_embed(f"{subtask}: {goal}", self.hash_dim)
        mapping = dict(mapping)
        mapping["condition_type"] = self.condition_type
        mapping["condition_dim"] = int(cond.shape[0])
        if self.condition_type == "sbert":
            mapping["condition_model"] = self.sbert_model_name
        self.current_mapping = mapping
        self.current_condition = cond.astype(np.float32)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)

    def _make_observation(self, obs: dict) -> dict[str, torch.Tensor]:
        if self.current_condition is None:
            raise RuntimeError("Model must be reset with a subtask before step()")
        state = np.concatenate([np.asarray(obs["robot_obs"], dtype=np.float32), self.current_condition])
        static_image = self._image_to_tensor(obs["rgb_obs"]["rgb_static"])
        wrist_image = self._image_to_tensor(obs["rgb_obs"]["rgb_gripper"])
        observation = {"observation.state": torch.from_numpy(state)}

        image_features = list(getattr(self.policy.config, "image_features", []) or [])
        if not image_features:
            image_features = ["observation.images.top", "observation.images.wrist"]

        for key in image_features:
            if key in {"observation.images.top", "observation.images.image"}:
                observation[key] = static_image
            elif key in {"observation.images.wrist", "observation.images.wrist_image"}:
                observation[key] = wrist_image
            else:
                raise KeyError(f"Unsupported ACT image feature key in checkpoint config: {key}")
        return observation

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
    parser.add_argument("--task-meta", default=str(base.DEFAULT_TASK_META))
    parser.add_argument("--run-id", default="calvin_conditioned_eval")
    parser.add_argument(
        "--condition-type",
        choices=["onehot", "zero_onehot", "wrong_onehot", "sbert", "hash"],
        default="onehot",
    )
    parser.add_argument("--sbert-model", default=os.environ.get("HW3_LANG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--hash-dim", type=int, default=384)
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
    task_meta = Path(args.task_meta).resolve()
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
        "task_meta": str(task_meta),
        "condition_type": args.condition_type,
        "sbert_model": args.sbert_model if args.condition_type == "sbert" else None,
        "num_sequences": args.num_sequences,
        "ep_len": args.ep_len,
        "completed_sequences": 0,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    base.write_status(status_file, status)
    env = base.make_env(dataset_path)
    task_oracle = base.make_task_oracle()
    val_annotations = base.load_validation_annotations()
    model = GenericConditionedACTCalvinModel(
        checkpoint, task_meta, args.device, args.condition_type, args.sbert_model, args.hash_dim
    )
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
        "condition_type": args.condition_type,
        "sbert_model": args.sbert_model if args.condition_type == "sbert" else None,
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
