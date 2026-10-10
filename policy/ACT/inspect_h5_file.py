import h5py
import argparse
import sys

def inspect_h5_file(file_path):
    """
    EnglishH5English
    """
    print(f"Inspecting H5 file: {file_path}")
    print("=" * 80)

    try:
        with h5py.File(file_path, "r") as f:
            def print_structure(name, obj):
                indent = "  " * name.count("/")
                print(f"\n{indent}{name}")

                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}  Type: Dataset")
                    print(f"{indent}  Shape: {obj.shape}")
                    print(f"{indent}  Dtype: {obj.dtype}")

                    # English
                    try:
                        if obj.size <= 10:
                            data = obj[()]
                            print(f"{indent}  Data: {data}")
                        else:
                            # English,English
                            data = obj[()]
                            if len(obj.shape) == 1:
                                print(f"{indent}  Data (first 10): {data[:10]}...")
                            else:
                                # English,English,English
                                print(f"{indent}  Data: (high-dimensional array, not printed)")
                    except Exception as e:
                        print(f"{indent}  Data: (cannot read: {str(e)})")

                elif isinstance(obj, h5py.Group):
                    print(f"{indent}  Type: Group")

                # English
                if len(obj.attrs) > 0:
                    print(f"{indent}  Attributes:")
                    for key, val in obj.attrs.items():
                        print(f"{indent}    {key}: {val}")

            f.visititems(print_structure)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect H5 file structure")
    parser.add_argument("file_path", type=str, help="Path to H5 file")
    args = parser.parse_args()

    inspect_h5_file(args.file_path)
