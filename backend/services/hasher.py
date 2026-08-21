import hashlib
from pathlib import Path
from typing import Union


def calculate_sha256_file(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Computes SHA-256 hash of a file on disk using streaming chunks.
    Guarantees O(1) memory usage regardless of video/file size.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")

    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
            
    return sha256_hash.hexdigest()


def calculate_sha256_bytes(data: bytes) -> str:
    """Computes SHA-256 hash directly from byte buffer."""
    return hashlib.sha256(data).hexdigest()


def verify_file_integrity(file_path: Union[str, Path], expected_hash: str) -> bool:
    """
    Evidence Guardian integrity check:
    Recomputes SHA-256 and checks against the original registered fingerprint.
    Returns True if authentic and untampered, False if modified.
    """
    actual_hash = calculate_sha256_file(file_path)
    return actual_hash.lower() == expected_hash.lower()
