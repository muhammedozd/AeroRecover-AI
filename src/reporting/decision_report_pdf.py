"""Generate downloadable AeroRecover AI decision reports."""

from datetime import datetime, timezone
from math import isfinite
from io import BytesIO
from reportlab.pdfgen.canvas import Canvas
import pandas as pd
from html import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.decision_support.contracts import (
    FlightDecisionInput,
    FlightDecisionReport,
)
from src.explainability.local_shap import LocalShapExplanation
from src.models.rotation_model_contract import MODEL_THRESHOLD, MODEL_VERSION

PDF_COLORS = {
    "navy": colors.HexColor("#071827"),
    "panel": colors.HexColor("#0E2A43"),
    "border": colors.HexColor("#245573"),
    "cyan": colors.HexColor("#38BDF8"),
    "text": colors.HexColor("#17212B"),
    "muted": colors.HexColor("#60788A"),
    "white": colors.HexColor("#FFFFFF"),
    "p1": colors.HexColor("#FF5C6C"),
    "p2": colors.HexColor("#FFAA4C"),
    "p3": colors.HexColor("#FFD166"),
    "p4": colors.HexColor("#48D597"),
}

REQUIRED_CHAIN_COLUMNS = {
    "SOURCE_FLIGHT_ID",
    "TARGET_FLIGHT_ID",
    "CONNECTION_AIRPORT",
    "PROPAGATION_PROBABILITY",
    "PLANNED_CONNECTION_MINUTES",
}


def build_pdf_styles() -> dict[str, ParagraphStyle]:
    
    base_styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            name="AeroRecoverTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=PDF_COLORS["navy"],
            alignment=TA_LEFT,
            spaceAfter=1.5 * mm,
        ),
        "subtitle": ParagraphStyle(
            name="AeroRecoverSubtitle",
            parent=base_styles["Heading2"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=PDF_COLORS["muted"],
            spaceAfter=4 * mm,
        ),
        "section": ParagraphStyle(
            name="AeroRecoverSection",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=PDF_COLORS["navy"],
            spaceBefore=4 * mm,
            spaceAfter=3 * mm,
        ),
        "body": ParagraphStyle(
            name="AeroRecoverBody",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=PDF_COLORS["text"],
            spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            name="AeroRecoverSmall",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=PDF_COLORS["muted"],
        ),
        "table": ParagraphStyle(
            name="AeroRecoverTable",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=6.5,
            leading=8,
            textColor=PDF_COLORS["text"],
        ),
        "table_header": ParagraphStyle(
            name="AeroRecoverTableHeader",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.3,
            leading=7.5,
            alignment=TA_CENTER,
            textColor=PDF_COLORS["white"],
        ),
        "warning": ParagraphStyle(
            name="AeroRecoverWarning",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#8A2630"),
            borderColor=PDF_COLORS["p1"],
            borderWidth=0.8,
            borderPadding=7,
            backColor=colors.HexColor("#FFF1F3"),
            spaceAfter=3 * mm,
        ),
        "card_label": ParagraphStyle(
            name="AeroRecoverCardLabel",
            parent=base_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9,
            textColor=PDF_COLORS["muted"],
            spaceAfter=0,
        ),
        "card_body": ParagraphStyle(
            name="AeroRecoverCardBody",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=9.4,
            textColor=PDF_COLORS["text"],
            spaceAfter=0,
        ),
    }

def draw_page_frame(
    pdf_canvas: Canvas,
    document: SimpleDocTemplate,
) -> None:
    page_width, page_height = A4

    pdf_canvas.saveState()

    pdf_canvas.setFillColor(
        PDF_COLORS["navy"]
    )
    pdf_canvas.rect(
        0,
        page_height - 22 * mm,
        page_width,
        22 * mm,
        fill=1,
        stroke=0,
    )

    pdf_canvas.setFillColor(
        PDF_COLORS["white"]
    )
    pdf_canvas.setFont(
        "Helvetica-Bold",
        10,
    )
    pdf_canvas.drawString(
        18 * mm,
        page_height - 13 * mm,
        "AERORECOVER AI",
    )

    pdf_canvas.setFont(
        "Helvetica",
        8,
    )
    pdf_canvas.drawRightString(
        page_width - 18 * mm,
        page_height - 13 * mm,
        "FLIGHT DECISION REPORT",
    )

    footer_y = 12 * mm

    pdf_canvas.setStrokeColor(
        PDF_COLORS["border"]
    )
    pdf_canvas.setLineWidth(0.5)
    pdf_canvas.line(
        18 * mm,
        footer_y + 4 * mm,
        page_width - 18 * mm,
        footer_y + 4 * mm,
    )

    pdf_canvas.setFillColor(
        PDF_COLORS["muted"]
    )
    pdf_canvas.setFont(
        "Helvetica",
        7,
    )
    pdf_canvas.drawString(
        18 * mm,
        footer_y,
        "Historical validation replay - not live operations",
    )

    pdf_canvas.drawRightString(
        page_width - 18 * mm,
        footer_y,
        f"Page {document.page}",
    )

    pdf_canvas.restoreState()


class AeroRecoverDocTemplate(SimpleDocTemplate):
    """Draw the fixed report frame after flowables so it remains visible."""

    def afterPage(self) -> None:
        draw_page_frame(self.canv, self)

def build_metric_table(
    metrics: list[tuple[str, str]],
    styles: dict[str, ParagraphStyle],
) -> Table:
    table_rows = []

    for index in range(
        0,
        len(metrics),
        2,
    ):
        metric_pair = metrics[
            index:index + 2
        ]

        table_row = []

        for label, value in metric_pair:
            metric_content = [
                Paragraph(
                    escape(label.upper()),
                    styles["small"],
                ),
                Spacer(
                    1,
                    1.5 * mm,
                ),
                Paragraph(
                    f"<b>{escape(value)}</b>",
                    styles["body"],
                ),
            ]

            table_row.append(
                metric_content
            )

        if len(table_row) == 1:
            table_row.append("")

        table_rows.append(
            table_row
        )

    metric_table = Table(
        table_rows,
        colWidths=[
            87 * mm,
            87 * mm,
        ],
        hAlign="LEFT",
        splitByRow=0,
    )

    metric_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#F3F7FA"),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                PDF_COLORS["border"],
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#CBD8E1"),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4 * mm,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                2.2 * mm,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                2.2 * mm,
            ),
        ])
    )

    return metric_table


def validate_predicted_chain(predicted_chain: pd.DataFrame) -> None:
    """Validate that the report has auditable edge-level chain evidence."""
    if predicted_chain.empty:
        raise ValueError("Cannot build a decision report from an empty predicted chain.")

    missing = REQUIRED_CHAIN_COLUMNS.difference(predicted_chain.columns)
    if missing:
        raise ValueError(
            "Predicted chain is missing required columns: "
            + ", ".join(sorted(missing))
        )


def build_chain_table(
    predicted_chain: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
) -> Table:
    headers = [
        "#", "Source flight", "Target flight", "Connection",
        "Edge probability", "Planned connection",
    ]
    rows = [[Paragraph(escape(label), styles["table_header"]) for label in headers]]
    for sequence, (_, edge) in enumerate(predicted_chain.iterrows(), start=1):
        rows.append([
            Paragraph(str(sequence), styles["table"]),
            Paragraph(escape(str(edge["SOURCE_FLIGHT_ID"])), styles["table"]),
            Paragraph(escape(str(edge["TARGET_FLIGHT_ID"])), styles["table"]),
            Paragraph(escape(str(edge["CONNECTION_AIRPORT"])), styles["table"]),
            Paragraph(f'{float(edge["PROPAGATION_PROBABILITY"]):.1%}', styles["table"]),
            Paragraph(f'{float(edge["PLANNED_CONNECTION_MINUTES"]):.0f} min', styles["table"]),
        ])

    table = Table(
        rows,
        colWidths=[8 * mm, 48 * mm, 48 * mm, 19 * mm, 25 * mm, 26 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PDF_COLORS["panel"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD8E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    return table


def build_recommendation_card(
    recommendation: dict[str, object],
    styles: dict[str, ParagraphStyle],
) -> KeepTogether:
    priority = recommendation["priority"]
    priority_name = getattr(priority, "name", str(priority)).replace("_", " ")
    label = styles["card_label"]
    body = styles["card_body"]
    rows = [
        [
            Paragraph("Action code", label),
            Paragraph(escape(str(recommendation["action_code"])), body),
            Paragraph("Priority", label),
            Paragraph(escape(priority_name), body),
        ],
        [Paragraph("Action", label), Paragraph(escape(str(recommendation["action"])), body), "", ""],
        [Paragraph("Reason", label), Paragraph(escape(str(recommendation["reason"])), body), "", ""],
        [
            Paragraph("Owner", label),
            Paragraph(escape(str(recommendation["owner"])), body),
            Paragraph("Timing", label),
            Paragraph(escape(str(recommendation["timing"])), body),
        ],
        [Paragraph("Objective", label), Paragraph(escape(str(recommendation["objective"])), body), "", ""],
        [
            Paragraph("Feasibility", label),
            Paragraph(escape(str(recommendation["feasibility_note"])), body),
            "",
            "",
        ],
    ]
    card = Table(rows, colWidths=[22 * mm, 66 * mm, 18 * mm, 68 * mm], hAlign="LEFT")
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7FA")),
        ("BOX", (0, 0), (-1, -1), 0.7, PDF_COLORS["border"]),
        ("LINEBEFORE", (0, 0), (0, -1), 3, PDF_COLORS["cyan"]),
        ("SPAN", (1, 1), (3, 1)),
        ("SPAN", (1, 2), (3, 2)),
        ("SPAN", (1, 4), (3, 4)),
        ("SPAN", (1, 5), (3, 5)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 0.55 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.55 * mm),
    ]))
    return KeepTogether([card, Spacer(1, 1 * mm)])


def build_shap_table(
    explanation: LocalShapExplanation,
    styles: dict[str, ParagraphStyle],
) -> Table:
    rows = [[
        Paragraph("Feature", styles["table_header"]),
        Paragraph("Value", styles["table_header"]),
        Paragraph("Contribution", styles["table_header"]),
        Paragraph("Direction", styles["table_header"]),
    ]]
    for row in explanation.contributions.head(8).itertuples():
        feature_value = f"{row.feature_value:.3f}" if isinstance(row.feature_value, float) else str(row.feature_value)
        rows.append([
            Paragraph(escape(str(row.feature)), styles["table"]),
            Paragraph(escape(feature_value), styles["table"]),
            Paragraph(f"{float(row.shap_value):+.5f}", styles["table"]),
            Paragraph(escape(str(row.direction)), styles["table"]),
        ])
    table = Table(
        rows,
        colWidths=[64 * mm, 42 * mm, 34 * mm, 34 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PDF_COLORS["panel"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD8E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4 * mm),
    ]))
    return table

def build_decision_report_pdf(
    *,
    flight_id: str,
    decision_input: FlightDecisionInput,
    decision_report: FlightDecisionReport,
    predicted_chain: pd.DataFrame,
    cumulative_chain_score: float,
    map_image_bytes: bytes | None = None,
    local_explanation: LocalShapExplanation | None = None,
    shap_image_bytes: bytes | None = None,
    shap_error_message: str | None = None,
) -> bytes:
    """Build an in-memory A4 report for one historical validation replay."""
    validate_predicted_chain(predicted_chain)
    if not isfinite(cumulative_chain_score):
        raise ValueError("Cumulative chain score must be finite.")

    pdf_buffer = BytesIO()

    document = AeroRecoverDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=30 * mm,
        bottomMargin=22 * mm,
        title="AeroRecover AI Flight Decision Report",
        author="AeroRecover AI",
    )

    styles = build_pdf_styles()

    generated_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    assessment = decision_report.assessment

    map_section = []
    if map_image_bytes is not None:
        map_image = Image(BytesIO(map_image_bytes))
        available_width = 174 * mm
        map_image.drawWidth = available_width
        map_image.drawHeight = available_width * (
            map_image.imageHeight / map_image.imageWidth
        )
        map_section = [
            CondPageBreak(108 * mm),
            Spacer(1, 7 * mm),
            KeepTogether([
                Paragraph("Predicted Propagation Map", styles["section"]),
                map_image,
                Spacer(1, 2 * mm),
                Paragraph(
                    "Map lines visualize coordinate interpolation for the model-predicted "
                    "propagation sequence. They are not recorded aircraft trajectories.",
                    styles["small"],
                ),
            ]),
        ]

    shap_section = []
    if local_explanation is not None and shap_image_bytes is not None:
        shap_image = Image(BytesIO(shap_image_bytes))
        shap_image.drawWidth = 174 * mm
        shap_image.drawHeight = shap_image.drawWidth * (
            shap_image.imageHeight / shap_image.imageWidth
        )
        shap_section = [
            CondPageBreak(82 * mm),
            KeepTogether([
                Paragraph("Local Model Explanation", styles["section"]),
                shap_image,
            ]),
            Paragraph(
                "SHAP contributions are expressed in the model's raw-score space. Positive "
                "values increase the predicted propagation score and negative values decrease "
                "it. These values do not establish causality.",
                styles["small"],
            ),
            build_metric_table(
                metrics=[
                    ("Model probability", f"{local_explanation.model_probability:.6f}"),
                    ("Model raw score", f"{local_explanation.model_raw_score:.6f}"),
                    ("SHAP raw score", f"{local_explanation.shap_raw_score:.6f}"),
                    ("Reconstruction error", f"{local_explanation.reconstruction_error:.2e}"),
                ],
                styles=styles,
            ),
            build_shap_table(local_explanation, styles),
        ]
    elif shap_error_message:
        shap_section = [
            CondPageBreak(28 * mm),
            KeepTogether([
                Paragraph("Local Model Explanation", styles["section"]),
                Paragraph(
                    "The local explanation could not be generated: "
                    + escape(shap_error_message),
                    styles["body"],
                ),
            ]),
        ]

    story = [
        Paragraph("AeroRecover AI", styles["title"]),
        Paragraph("Historical Validation Decision-Support Report", styles["subtitle"]),
        Paragraph(
            "NOT LIVE OPERATIONS - Historical validation decision-support output only.",
            styles["warning"],
        ),
        Paragraph("Report metadata", styles["section"]),
        build_metric_table(
            metrics=[
                ("Selected start flight", str(flight_id)),
                ("Generated at", generated_at),
                ("Data period", "September-October 2023 validation"),
                ("Model version", MODEL_VERSION),
                ("Operational alert threshold", f"{MODEL_THRESHOLD:.2f}"),
                ("Operational status", "Historical replay - not live operations"),
            ],
            styles=styles,
        ),
        Paragraph("Decision Support Assessment", styles["section"]),
        build_metric_table(
            metrics=[
                ("Propagation probability", f"{decision_input.propagation_probability:.1%}"),
                ("Operational priority", assessment.priority.name.replace("_", " ")),
                ("Likelihood", assessment.likelihood.value.replace("_", " ")),
                ("Impact", assessment.impact.value.replace("_", " ")),
                ("Urgency", assessment.urgency.value.replace("_", " ")),
            ],
            styles=styles,
        ),
        Paragraph("Rotation Conditions", styles["section"]),
        build_metric_table(
            metrics=[
                ("Previous arrival delay", f"{decision_input.previous_arrival_delay:.1f} min"),
                ("Turn buffer", f"{decision_input.turn_buffer:.1f} min"),
                ("Previous delay ratio", f"{decision_input.previous_delay_ratio:.3f}"),
                ("Planned turnaround", f"{decision_input.planned_turnaround:.1f} min"),
            ],
            styles=styles,
        ),
        KeepTogether([
            Paragraph("Predicted Domino Chain Summary", styles["section"]),
            build_metric_table(
                metrics=[
                    ("Edge count", str(len(predicted_chain))),
                    ("Flight count", str(len(predicted_chain) + 1)),
                    ("Cumulative chain score", f"{cumulative_chain_score:.1%}"),
                ],
                styles=styles,
            ),
            Paragraph(
                "The cumulative chain score compounds model outputs across the displayed chain; "
                "it is not a calibrated real-world end-to-end chain probability.",
                styles["small"],
            ),
        ]),
        *shap_section,
        *map_section,
        Paragraph("Edge-level Domino Chain", styles["section"]),
        build_chain_table(predicted_chain, styles),
    ]

    recommendation_cards = [
        build_recommendation_card(recommendation, styles)
        for recommendation in decision_report.recommendations
    ]
    if recommendation_cards:
        story.extend([
            CondPageBreak(48 * mm),
            KeepTogether([
                Paragraph("Operational Recommendations", styles["section"]),
                recommendation_cards[0],
            ]),
            *recommendation_cards[1:],
        ])

    story.extend([
        CondPageBreak(24 * mm),
        KeepTogether([
            Paragraph("Interpretation and Limitations", styles["section"]),
            Paragraph(
                "This system is not a live operations system. Map movement is coordinate "
                "interpolation and is not a recorded real-flight trajectory. Results describe "
                "predictive associations and do not establish causality.",
                styles["body"],
            ),
        ]),
        Paragraph(
            "Recommendations support human judgment and are not automatic operational commands. "
            "Live weather, gate, crew, maintenance, passenger, and station-resource information "
            "is not available. The cumulative chain score must not be interpreted as a calibrated "
            "real-world chain probability.",
            styles["body"],
        ),
    ])

    document.build(story)

    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()

    return pdf_bytes
