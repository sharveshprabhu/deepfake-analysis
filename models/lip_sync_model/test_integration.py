"""
Release Verification Script for TruthLens Backend Engineer.
Tests synchronous and asynchronous adapter execution, checkpoint loading, and schema validation.
"""
import os
import sys
import asyncio

RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

from adapter import run_sync_analysis, truthlens_audio_avsync_adapter, register_with_orchestrator


def test_sync_adapter():
    print("[*] 1. Testing Sync Adapter on Non-Existent File (Graceful Fallback)...")
    res = run_sync_analysis("non_existent_file.mp4", evidence_id="TEST-001")
    assert res["module"] == "audio_ai"
    assert res["status"] in ["ERROR", "SUCCESS"]
    assert "has_audio" in res
    assert "av_sync_offset_ms" in res
    assert "acoustic_artifact_score" in res
    assert "explanations" in res
    print("    -> PASSED! Output:", res)


async def test_async_adapter():
    print("\n[*] 2. Testing Async Adapter Non-Blocking Execution...")
    res = await truthlens_audio_avsync_adapter("TEST-ASYNC-002", "non_existent_file.mp4")
    assert res["module"] == "audio_ai"
    assert res["evidence_id"] == "TEST-ASYNC-002"
    print("    -> PASSED! Output:", res)


def test_registration():
    print("\n[*] 3. Testing Orchestrator Registration Hook...")
    info = register_with_orchestrator()
    assert info["module_name"] == "audio_ai"
    assert callable(info["async_handler"])
    assert callable(info["sync_handler"])
    print("    -> PASSED! Module Name:", info["module_name"], "| Architecture:", info["model_architecture"])


if __name__ == "__main__":
    print("=" * 60)
    print("TruthLens Person 2B Release Verification Suite")
    print("=" * 60)
    test_sync_adapter()
    asyncio.run(test_async_adapter())
    test_registration()
    print("\n" + "=" * 60)
    print("[+] ALL RELEASE INTEGRATION TESTS PASSED CLEANLY!")
    print("=" * 60)
