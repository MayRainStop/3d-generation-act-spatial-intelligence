from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collect_results", ROOT / "scripts" / "06_collect_results.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_parse_losses_from_common_log_formats(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "step=1 loss=2.5",
                "step=2 action_l1=1.75",
                "step=3 l1_loss: 1.25",
            ]
        ),
        encoding="utf-8",
    )
    assert module.parse_losses(log) == [2.5, 1.75, 1.25]
