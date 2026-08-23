from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments/end2end_memory_systems_20260804"
    / "systems_v165_plant_seedlings_transfer_attributes_a100_6h_luna_low"
    / "preflight_public_pretrained.py"
)


def load_module():
    spec = spec_from_file_location("public_pretrained_preflight_v165", SCRIPT)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_repository_cache_dir_matches_huggingface_layout(tmp_path):
    module = load_module()
    assert module.repository_cache_dir(tmp_path, module.HF_REPOSITORY) == (
        tmp_path / "models--timm--efficientnet_b3.ra2_in1k"
    )


def test_current_snapshot_rejects_unsafe_revision(tmp_path):
    module = load_module()
    repository = tmp_path / "repo"
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text("../escape")
    try:
        module.current_snapshot(repository)
    except RuntimeError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("unsafe Hub revision must be rejected")


def test_current_snapshot_resolves_existing_revision(tmp_path):
    module = load_module()
    repository = tmp_path / "repo"
    snapshot = repository / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("abc123\n")
    assert module.current_snapshot(repository) == snapshot


def test_receipt_does_not_require_unused_bin_cache_name(tmp_path):
    module = load_module()
    assert module.optional_file_receipt(tmp_path / "pytorch_model.bin") is None
