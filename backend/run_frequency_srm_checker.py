"""
Standalone CLI tool to run the TruthLens Frequency & SRM Inconsistency Diagnostic Worker.
Usage:
    python run_frequency_srm_checker.py
    python run_frequency_srm_checker.py --image path/to/image.jpg
"""

import sys
import argparse
from pathlib import Path

# Add backend directory to sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from backend.services.frequency_srm_worker import global_frequency_srm_worker


def main():
    parser = argparse.ArgumentParser(description="TruthLens Frequency & SRM Inconsistency Diagnostic Worker")
    parser.add_argument("--image", type=str, default=None, help="Optional path to a specific image file to audit")
    args = parser.parse_args()

    sample_paths = [args.image] if args.image else None
    report = global_frequency_srm_worker.run_comprehensive_audit(sample_paths=sample_paths)

    print("\n" + "=" * 70)
    print(" 🔬 TRUTHLENS FREQUENCY & SRM INCONSISTENCY FORENSIC AUDIT")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total Samples Audited: {report['total_samples_audited']}\n")

    print("📊 KEY FINDINGS & MATHEMATICAL EXPLANATION:")
    print("-" * 70)
    print(f"Why calibrated scores are < 1.0:\n  {report['key_findings']['why_calibrated_score_is_less_than_1']}\n")
    print(f"Do raw physical metrics exceed 1.0:\n  {report['key_findings']['do_raw_physical_metrics_exceed_1']}\n")

    print("📋 SAMPLE-BY-SAMPLE DEEP AUDIT TABLE:")
    print("-" * 70)
    print(f"{'Sample Name':<20} | {'Raw P90/P10':<12} | {'Raw CV':<8} | {'SRM Score':<10} | {'Freq Score':<10} | {'Raw > 1?':<8}")
    print("-" * 70)

    for sample in report["sample_audits"]:
        fn = sample["filename"]
        summ = sample["summary"]
        p90_p10 = f"{summ['raw_p90_over_p10_ratio']:.2f}" if summ['raw_p90_over_p10_ratio'] is not None else "N/A"
        cv = f"{summ['raw_cv']:.2f}" if summ['raw_cv'] is not None else "N/A"
        srm = f"{summ['calibrated_srm_score']:.3f}" if summ['calibrated_srm_score'] is not None else "N/A"
        freq = f"{summ['calibrated_frequency_score']:.3f}" if summ['calibrated_frequency_score'] is not None else "N/A"
        raw_gt_1 = "YES (P90/P10)" if summ.get("does_raw_dynamic_range_exceed_1") else "NO"

        print(f"{fn:<20} | {p90_p10:<12} | {cv:<8} | {srm:<10} | {freq:<10} | {raw_gt_1:<8}")

    print("=" * 70)
    print("✅ Full telemetry report persisted to storage/reports/frequency_srm_audit.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
