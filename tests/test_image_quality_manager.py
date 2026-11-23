from pathlib import Path

from image_quality_manager import ImageQualityManager


def test_quality_manager_ignores_filename_similarity_without_equal_bytes(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "portrait.jpg").write_bytes(b"first image")
    (image_dir / "portrait-300x200.jpg").write_bytes(b"second image")

    manager = ImageQualityManager(image_dir, tmp_path / "metadata.json")

    assert manager.find_all_duplicates() == {}


def test_quality_manager_groups_identical_files_with_different_names(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "portrait.jpg").write_bytes(b"identical bytes")
    (image_dir / "renamed.png").write_bytes(b"identical bytes")

    manager = ImageQualityManager(image_dir, tmp_path / "metadata.json")
    groups = manager.find_all_duplicates()

    assert len(groups) == 1
    keep, duplicates = next(iter(groups.values()))
    assert Path(keep).is_file()
    assert len(duplicates) == 1
