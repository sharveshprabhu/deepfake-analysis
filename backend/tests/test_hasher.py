import pytest
from pathlib import Path
from backend.services.hasher import calculate_sha256_file, calculate_sha256_bytes, verify_file_integrity


def test_sha256_bytes():
    data = b"TruthLens Forensic Hashing"
    h1 = calculate_sha256_bytes(data)
    assert len(h1) == 64
    assert h1 == calculate_sha256_bytes(data)


def test_sha256_file(tmp_path):
    test_file = tmp_path / "test_media.bin"
    test_file.write_bytes(b"Simulated Deepfake Video Bytes")

    digest = calculate_sha256_file(test_file)
    assert len(digest) == 64
    assert verify_file_integrity(test_file, digest) is True

    # Tamper with file
    test_file.write_bytes(b"Tampered Deepfake Video Bytes")
    assert verify_file_integrity(test_file, digest) is False
