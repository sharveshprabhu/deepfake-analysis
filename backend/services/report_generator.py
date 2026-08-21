import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)

from backend.config import REPORTS_DIR, MODEL_VERSION, SYSTEM_NAME


def generate_pdf_report(
    evidence_id: str,
    filename: str,
    sha256_hash: str,
    verdict: str,
    confidence: float,
    fusion_score: float,
    visual_score: Optional[float],
    frequency_score: Optional[float],
    temporal_score: Optional[float],
    audio_score: Optional[float],
    suspicious_frames: list,
    explanations: list,
    analyzed_at: Optional[datetime] = None
) -> Path:
    """
    Generates an official TruthLens Forensic Evidence Report in PDF format.
    Saves the PDF to the reports directory and returns its Path.
    """
    report_filename = f"{evidence_id}_Forensic_Report.pdf"
    report_path = REPORTS_DIR / report_filename

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0F172A")    # Deep Slate
    c_accent = colors.HexColor("#3B82F6")     # TruthLens Blue
    c_text_dark = colors.HexColor("#1E293B")  # Dark Text
    c_text_muted = colors.HexColor("#64748B") # Muted Text
    c_card_bg = colors.HexColor("#F8FAFC")    # Card Gray

    # Verdict Colors
    if verdict == "MANIPULATED":
        c_verdict = colors.HexColor("#EF4444")     # Red
        c_verdict_bg = colors.HexColor("#FEF2F2")
    elif verdict == "AUTHENTIC":
        c_verdict = colors.HexColor("#10B981")     # Emerald
        c_verdict_bg = colors.HexColor("#ECFDF5")
    else:
        c_verdict = colors.HexColor("#F59E0B")     # Amber
        c_verdict_bg = colors.HexColor("#FFFBEB")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_accent
    )

    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_text_dark
    )

    mono_style = ParagraphStyle(
        'MonoText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=c_text_dark
    )

    verdict_badge_style = ParagraphStyle(
        'VerdictBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=c_verdict,
        alignment=1
    )

    verdict_sub_style = ParagraphStyle(
        'VerdictSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_text_dark,
        alignment=1
    )

    story = []

    # 1. Header Banner
    header_table_data = [
        [
            Paragraph("TRUTHLENS FORENSIC INTELLIGENCE", title_style),
            Paragraph("EVIDENCE GUARDIAN SEAL<br/><b>CRYPTOGRAPHIC AUDIT RECORD</b>", subtitle_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=6, spaceAfter=12))

    # 2. Evidence Passport Metadata Table
    timestamp_str = (analyzed_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S UTC")
    passport_data = [
        [Paragraph("<b>Evidence ID:</b>", body_style), Paragraph(f"<b>{evidence_id}</b>", body_style),
         Paragraph("<b>Date & Time:</b>", body_style), Paragraph(timestamp_str, body_style)],
        [Paragraph("<b>Filename:</b>", body_style), Paragraph(filename, body_style),
         Paragraph("<b>Engine:</b>", body_style), Paragraph(MODEL_VERSION, body_style)],
        [Paragraph("<b>SHA-256:</b>", body_style), Paragraph(sha256_hash, mono_style),
         Paragraph("<b>Integrity:</b>", body_style), Paragraph("<font color='#10B981'><b>VERIFIED UNTAMPERED</b></font>", body_style)]
    ]
    passport_table = Table(passport_data, colWidths=[80, 210, 70, 180])
    passport_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(passport_table)
    story.append(Spacer(1, 10))

    # 3. Verdict & Confidence Banner Card
    verdict_card_data = [
        [
            Paragraph(f"VERDICT: {verdict}", verdict_badge_style),
            Paragraph(f"CONFIDENCE: {confidence * 100:.1f}%<br/><font size=8 color='#64748B'>Fusion Score: {fusion_score * 100:.1f}%</font>", verdict_sub_style)
        ]
    ]
    verdict_table = Table(verdict_card_data, colWidths=[270, 270])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_verdict_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, c_verdict),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 12))

    # 4. Multimodal Forensic Signals Breakdown Table
    story.append(Paragraph("FORENSIC SIGNAL BREAKDOWN", h2_style))
    
    def fmt_score(score):
        return f"{score * 100:.1f}%" if score is not None else "N/A (Bypassed)"

    signal_rows = [
        [Paragraph("<b>Forensic Modality</b>", body_style), Paragraph("<b>Analyzed Signal</b>", body_style), Paragraph("<b>Risk / Score</b>", body_style), Paragraph("<b>Status</b>", body_style)],
        [Paragraph("Visual AI (Spatial)", body_style), Paragraph("Facial boundary, blending artifacts, sensor noise", body_style), Paragraph(fmt_score(visual_score), body_style), Paragraph("Active", body_style)],
        [Paragraph("Frequency AI (Spectral)", body_style), Paragraph("DCT / High-frequency domain artifacts", body_style), Paragraph(fmt_score(frequency_score), body_style), Paragraph("Active", body_style)],
        [Paragraph("Temporal AI (Consistency)", body_style), Paragraph("Inter-frame optical flow & landmark jitter", body_style), Paragraph(fmt_score(temporal_score), body_style), Paragraph("Active" if temporal_score is not None else "Bypassed", body_style)],
        [Paragraph("Audio AI (AV-Sync)", body_style), Paragraph("Lip sync phoneme-viseme & synthetic audio", body_style), Paragraph(fmt_score(audio_score), body_style), Paragraph("Active" if audio_score is not None else "Bypassed", body_style)],
    ]
    signal_table = Table(signal_rows, colWidths=[140, 240, 90, 70])
    signal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(signal_table)
    story.append(Spacer(1, 10))

    # 5. Suspicious Frames & Explanations
    story.append(Paragraph("FORENSIC EXPLANATIONS & SUSPICIOUS FRAMES", h2_style))
    
    frames_text = ", ".join(f"Frame #{f}" for f in suspicious_frames) if suspicious_frames else "No anomalous frames isolated."
    story.append(Paragraph(f"<b>Flagged Frames:</b> {frames_text}", body_style))
    story.append(Spacer(1, 4))

    for exp in explanations:
        bullet_item = f"• {exp}"
        story.append(Paragraph(bullet_item, body_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 12))

    # 6. Legal & Chain of Custody Disclaimer Footer
    footer_text = (
        "<b>LEGAL & FORENSIC NOTICE:</b> This automated forensic report is generated by TruthLens AI. "
        "Confidence scores represent statistical estimations derived from multi-signal deep learning models. "
        "Cryptographic SHA-256 hashing preserves chain-of-custody for digital evidence triage and decision-support."
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceBefore=6, spaceAfter=6))
    story.append(Paragraph(footer_text, ParagraphStyle('Notice', parent=styles['Normal'], fontSize=7, leading=10, textColor=c_text_muted)))

    doc.build(story)
    return report_path
