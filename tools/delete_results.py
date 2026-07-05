"""
* complete results.txt 빼고 다 지우는 코드
"""


from pathlib import Path
import shutil

# Adjust these bounds to select which epochs to clean.
start = 24
end = 27



BASE_DIR = Path("/home/user/heejun/L4DR/logs/Simple_Voxel_Encoder_multi_class/test_train_simple_voxel_encoder_sedan/test_kitti") # test_kitti 폴더 경로 넣기


def purge_run_dir(run_dir: Path) -> None:
    """Remove everything in a run directory except complete_results.txt."""
    if not run_dir.is_dir():
        return

    for entry in run_dir.iterdir():
        if entry.name == "complete_results.txt":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
            print(f"Removed directory: {entry}")
        else:
            entry.unlink()
            print(f"Removed file: {entry}")


def clean_epoch_dir(epoch_dir: Path) -> None:
    """Clean all run subdirectories (e.g., 0.1, 0.2, 0.3) for one epoch."""
    if not epoch_dir.exists():
        print(f"Missing epoch directory, skipping: {epoch_dir}")
        return

    for run_dir in epoch_dir.iterdir():
        purge_run_dir(run_dir)


def main() -> None:
    if not BASE_DIR.exists():
        print(f"Base path not found: {BASE_DIR}")
        return

    descending = start >= end
    epoch_range = (
        range(start, end - 1, -1) if descending else range(start, end + 1)
    )

    for epoch in epoch_range:
        epoch_dir = BASE_DIR / f"epoch_{epoch}_total"
        print(f"Cleaning: {epoch_dir}")
        clean_epoch_dir(epoch_dir)


if __name__ == "__main__":
    main()
