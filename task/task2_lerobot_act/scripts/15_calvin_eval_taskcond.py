#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CALVIN_ROOT = ROOT / "data" / "calvin_raw" / "calvin"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "20260601_task_conditioned"
    / "act_abc_taskcond_seed0"
    / "checkpoints"
    / "050000"
    / "pretrained_model"
)
DEFAULT_TASK_META = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "lerobot"
    / "fywang"
    / "calvin-task-ABC-D-lerobot"
    / "meta"
    / "tasks.parquet"
)


try:
    import pyhash

    _HASHER = pyhash.fnv1_32()
except Exception:
    import zlib

    _HASHER = lambda text: zlib.crc32(str(text).encode("utf-8"))


@contextmanager
def temp_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def patch_numpy_legacy_aliases() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def add_calvin_paths() -> None:
    paths = [
        CALVIN_ROOT / "calvin_env" / "tacto",
        CALVIN_ROOT / "calvin_models",
        CALVIN_ROOT / "calvin_env",
    ]
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def patch_calvin_git_hash_for_tarball(play_table_env_module) -> None:
    original_get_git_commit_hash = play_table_env_module.get_git_commit_hash

    def safe_get_git_commit_hash(repo_path: Path) -> str:
        try:
            return original_get_git_commit_hash(repo_path)
        except Exception:
            return "tarball-no-git-metadata"

    play_table_env_module.get_git_commit_hash = safe_get_git_commit_hash


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().replace("_", " ").split())


class TaskIndexResolver:
    def __init__(self, task_meta_path: Path):
        table = pd.read_parquet(task_meta_path)
        self.task_dim = int(table["task_index"].max()) + 1
        self.entries = []
        self.by_subtask: dict[str, list[dict]] = {}
        for full_task, row in table.sort_values("task_index").iterrows():
            if ":" in full_task:
                subtask, language = full_task.split(":", 1)
            else:
                subtask, language = full_task, full_task
            entry = {
                "task_index": int(row["task_index"]),
                "full_task": str(full_task),
                "subtask": subtask.strip(),
                "language": language.strip(),
                "norm_language": normalize_text(language),
            }
            self.entries.append(entry)
            self.by_subtask.setdefault(entry["subtask"], []).append(entry)

    def resolve(self, subtask: str, goal: str) -> dict:
        candidates = self.by_subtask.get(subtask)
        if not candidates:
            raise KeyError(f"Subtask {subtask!r} not found in training task metadata")

        norm_goal = normalize_text(goal)
        best = max(
            candidates,
            key=lambda item: SequenceMatcher(None, norm_goal, item["norm_language"]).ratio(),
        )
        score = SequenceMatcher(None, norm_goal, best["norm_language"]).ratio()
        return {
            "subtask": subtask,
            "goal": goal,
            "task_index": best["task_index"],
            "matched_training_task": best["full_task"],
            "match_score": score,
        }


class TaskConditionedACTCalvinModel:
    def __init__(self, checkpoint: Path, task_meta: Path, device: str):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        self.device = torch.device(device)
        self.resolver = TaskIndexResolver(task_meta)
        self.policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            checkpoint,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
        )
        self.current_mapping: dict | None = None

    def reset(self, subtask: str, goal: str) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.current_mapping = self.resolver.resolve(subtask, goal)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array)

    def _make_observation(self, obs: dict) -> dict[str, torch.Tensor]:
        if self.current_mapping is None:
            raise RuntimeError("Model must be reset with a subtask before step()")
        one_hot = np.zeros(self.resolver.task_dim, dtype=np.float32)
        one_hot[int(self.current_mapping["task_index"])] = 1.0
        state = np.concatenate([np.asarray(obs["robot_obs"], dtype=np.float32), one_hot])
        return {
            "observation.images.top": self._image_to_tensor(obs["rgb_obs"]["rgb_static"]),
            "observation.images.wrist": self._image_to_tensor(obs["rgb_obs"]["rgb_gripper"]),
            "observation.state": torch.from_numpy(state),
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


def count_success(results: list[int]) -> list[float]:
    if not results:
        return [0.0] * 5
    counter = Counter(results)
    return [sum(counter[j] for j in range(i, 6)) / len(results) for i in range(1, 6)]


def write_status(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_official_get_sequences():
    import types

    module_name = "calvin_agent.evaluation.utils"
    previous_module = sys.modules.get(module_name)
    fake_utils = types.ModuleType(module_name)
    fake_utils.temp_seed = temp_seed
    sys.modules[module_name] = fake_utils
    try:
        from calvin_agent.evaluation.multistep_sequences import get_sequences

        return get_sequences
    finally:
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        else:
            sys.modules.pop(module_name, None)


def get_env_state_for_initial_condition(initial_condition):
    pi = np.pi
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = _HASHER(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def make_env(dataset_path: Path):
    from calvin_env.envs import play_table_env

    patch_calvin_git_hash_for_tarball(play_table_env)
    validation = dataset_path / "validation"
    if not validation.exists():
        raise FileNotFoundError(f"Missing validation folder: {validation}")
    return play_table_env.get_env(validation, show_gui=False)


def make_task_oracle():
    import hydra
    from omegaconf import OmegaConf

    conf_dir = CALVIN_ROOT / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml")
    return hydra.utils.instantiate(task_cfg)


def load_validation_annotations():
    from omegaconf import OmegaConf

    conf_dir = CALVIN_ROOT / "calvin_models" / "conf"
    return OmegaConf.load(conf_dir / "annotations" / "new_playtable_validation.yaml")


def evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, ep_len, debug):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    subtask_logs = []

    for subtask in eval_sequence:
        obs = env.get_obs()
        lang_annotation = str(val_annotations[subtask][0])
        model.reset(subtask, lang_annotation)
        start_info = env.get_info()
        success = False
        steps_used = 0

        for step in range(ep_len):
            action = model.step(obs)
            obs, _, _, current_info = env.step(action)
            steps_used = step + 1
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                success = True
                break
            if debug and step % 25 == 0:
                print(f"  {subtask}: step {step}, action={np.round(action, 3).tolist()}", flush=True)

        subtask_logs.append(
            {
                "subtask": subtask,
                "language": lang_annotation,
                "success": success,
                "steps": steps_used,
                "mapping": model.current_mapping,
            }
        )
        if success:
            success_counter += 1
        else:
            break
    return success_counter, subtask_logs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--task-meta", default=str(DEFAULT_TASK_META))
    parser.add_argument("--run-id", default="calvin_taskcond_eval")
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--sequence-workers", type=int, default=1)
    parser.add_argument("--ep-len", type=int, default=360)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-log-dir", default=str(ROOT / "results" / "calvin_eval_taskcond"))
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    patch_numpy_legacy_aliases()
    add_calvin_paths()

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
        "num_sequences": args.num_sequences,
        "ep_len": args.ep_len,
        "completed_sequences": 0,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_status(status_file, status)

    env = make_env(dataset_path)
    task_oracle = make_task_oracle()
    val_annotations = load_validation_annotations()
    model = TaskConditionedACTCalvinModel(checkpoint, task_meta, args.device)

    get_sequences = load_official_get_sequences()
    eval_sequences = get_sequences(args.num_sequences, num_workers=args.sequence_workers)
    results: list[int] = []
    details = []
    status["status"] = "running"
    status["total_sequences"] = len(eval_sequences)
    write_status(status_file, status)

    try:
        for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences, start=1):
            seq_start = time.time()
            result, subtask_logs = evaluate_sequence(
                env,
                model,
                task_oracle,
                initial_state,
                eval_sequence,
                val_annotations,
                args.ep_len,
                args.debug,
            )
            results.append(int(result))
            detail = {
                "sequence_index": seq_idx,
                "initial_state": dict(initial_state),
                "eval_sequence": list(eval_sequence),
                "successful_subtasks": int(result),
                "duration_sec": time.time() - seq_start,
                "subtasks": subtask_logs,
            }
            details.append(detail)
            success_rates = count_success(results)
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
                write_status(status_file, status)
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
        write_status(status_file, status)
        detail_file.write_text(json.dumps(details, indent=2), encoding="utf-8")
        raise

    summary = {
        "run_id": args.run_id,
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint),
        "num_sequences": len(eval_sequences),
        "ep_len": args.ep_len,
        "avg_successful_sequence_len": float(np.mean(results)) if results else 0.0,
        "success_rates": {str(i + 1): count_success(results)[i] for i in range(5)},
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
    write_status(status_file, status)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
