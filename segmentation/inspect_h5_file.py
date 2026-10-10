import h5py
import argparse
import sys

def inspect_h5_file(file_path):
    """
    EnglishH5English
    """
    print(f"Inspecting H5 file: {file_path}")
    print("=" * 80)

    with h5py.File(file_path, "r") as f:
        print(f["observation"].keys())
        print("=" * 80)
        print(f["observation/front_camera"].keys())
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect H5 file structure")
    parser.add_argument("file_path", type=str, help="Path to H5 file")
    args = parser.parse_args()

    inspect_h5_file(args.file_path)
