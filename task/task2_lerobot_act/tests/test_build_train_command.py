from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_train_command", ROOT / "scripts" / "04_build_train_command.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_builds_policy_type_command_for_modern_help():
    row = {
        "run_id": "debug_smoke_seed0",
        "dataset_repo": "fywang/calvin-debug-lerobot",
        "output_dir": "outputs/debug",
        "chunk_size": "10",
        "n_action_steps": "10",
        "seed": "0",
        "batch_size": "4",
        "learning_rate": "0.0001",
        "steps": "200",
    }
    help_output = "--policy.type --dataset.repo_id --output_dir --seed --steps --batch_size --lr --policy.chunk_size --policy.n_action_steps --policy.push_to_hub --wandb.mode"
    cmd = module.build_command(row, help_output)
    assert "--policy.type=act" in cmd
    assert "--dataset.repo_id=fywang/calvin-debug-lerobot" in cmd
    assert "--policy.chunk_size=10" in cmd
    assert "--policy.n_action_steps=10" in cmd
    assert "--policy.push_to_hub=false" in cmd
    assert "--wandb.mode=offline" in cmd


def test_builds_policy_command_for_act_docs_style_help():
    row = {
        "run_id": "debug_smoke_seed0",
        "dataset_repo": "fywang/calvin-debug-lerobot",
        "output_dir": "outputs/debug",
        "chunk_size": "10",
        "n_action_steps": "10",
        "seed": "0",
        "batch_size": "4",
        "learning_rate": "0.0001",
        "steps": "200",
    }
    help_output = "--policy --dataset.repo_id --training.batch_size --training.lr --training.steps --policy.chunk_size --policy.push_to_hub"
    cmd = module.build_command(row, help_output)
    assert "--policy=act" in cmd
    assert "--training.batch_size=4" in cmd
    assert "--training.lr=0.0001" in cmd
    assert "--training.steps=200" in cmd
    assert "--policy.push_to_hub=false" in cmd
