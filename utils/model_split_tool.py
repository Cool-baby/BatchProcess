# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/10/4 12:45
@Description:   Tools for splitting, merging, and verifying large model files.
                Usage:
                    Split:
                        python model_split_tool.py split <input_file> <output_prefix>

                    Merge:
                        python model_split_tool.py merge <part1> <part2> ... <output_file>

                    Verify:
                        python model_split_tool.py verify <original_file> <merged_file>
"""
import os
import sys
import hashlib


# ==================== Split ====================

def split_file(input_file, output_prefix, max_size_gb=1.8):
    """Split a large file into multiple binary parts."""
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    max_size = int(max_size_gb * 1024 ** 3)
    file_size = os.path.getsize(input_file)

    print(f"Input file : {input_file}")
    print(f"File size  : {file_size / 1024 ** 3:.3f} GiB")
    print(f"Part size  : {max_size / 1024 ** 3:.3f} GiB")

    total_size = 0
    part_index = 1

    with open(input_file, "rb") as src:
        while True:
            data = src.read(max_size)
            if not data:
                break

            output_file = f"{output_prefix}.part{part_index}"
            with open(output_file, "wb") as dst:
                dst.write(data)

            part_size = os.path.getsize(output_file)
            total_size += part_size
            print(f"Created: {output_file} ({part_size / 1024 ** 3:.3f} GiB)")
            part_index += 1

    if total_size != file_size:
        raise RuntimeError("Split failed: file sizes do not match.")

    print(f"Split completed: {part_index - 1} parts.")


# ==================== Merge ====================

def merge_files(part_files, output_file):
    """Merge binary parts into the original file."""
    for part_file in part_files:
        if not os.path.isfile(part_file):
            raise FileNotFoundError(f"File not found: {part_file}")

    buffer_size = 64 * 1024 * 1024
    total_size = 0

    print("Merging:")
    with open(output_file, "wb") as dst:
        for part_file in part_files:
            part_size = os.path.getsize(part_file)
            print(f"  {part_file} ({part_size / 1024 ** 3:.3f} GiB)")

            with open(part_file, "rb") as src:
                while True:
                    data = src.read(buffer_size)
                    if not data:
                        break
                    dst.write(data)
                    total_size += len(data)

    print(f"Merged file: {output_file}")
    print(f"Output size: {total_size / 1024 ** 3:.3f} GiB")


# ==================== Verify ====================

def calculate_sha256(filename, buffer_size=64 * 1024 * 1024):
    """Calculate SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()

    with open(filename, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def verify_files(original_file, merged_file):
    """Verify whether two files are identical."""
    if not os.path.isfile(original_file):
        raise FileNotFoundError(f"File not found: {original_file}")
    if not os.path.isfile(merged_file):
        raise FileNotFoundError(f"File not found: {merged_file}")

    original_size = os.path.getsize(original_file)
    merged_size = os.path.getsize(merged_file)

    print(f"Original size: {original_size / 1024 ** 3:.3f} GiB")
    print(f"Merged size  : {merged_size / 1024 ** 3:.3f} GiB")

    if original_size != merged_size:
        print("Verification failed: file sizes differ.")
        return False

    print("Calculating SHA-256...")
    original_hash = calculate_sha256(original_file)
    merged_hash = calculate_sha256(merged_file)

    print(f"Original SHA-256: {original_hash}")
    print(f"Merged SHA-256  : {merged_hash}")

    if original_hash == merged_hash:
        print("Verification successful: files are identical.")
        return True

    print("Verification failed: files are different.")
    return False


# ==================== Main ====================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "split":
        if len(sys.argv) != 4:
            print("Usage: python model_split_tool.py split <input_file> <output_prefix>")
            return

        split_file(sys.argv[2], sys.argv[3])

    elif command == "merge":
        if len(sys.argv) < 5:
            print("Usage: python model_split_tool.py merge <part1> <part2> ... <output_file>")
            return

        merge_files(sys.argv[2:-1], sys.argv[-1])

    elif command == "verify":
        if len(sys.argv) != 4:
            print("Usage: python model_split_tool.py verify <original_file> <merged_file>")
            return

        verify_files(sys.argv[2], sys.argv[3])

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
