from __future__ import annotations

from app.engine.stockfish_engine import AnalysisResult

BEST_MOVE_HEADER = "BEST MOVE"
ALTERNATIVES_HEADER = "ALTERNATIVES"
PV_HEADER = "PRINCIPAL VARIATION"
ENGINE_HEADER = "--- ENGINE DATA ---"

_LABEL_WIDTH = 15


def format_score(result: AnalysisResult) -> str:
    """Evaluation from the side to move: +0.84, -0.50, M5, M-3."""
    if result.is_mate:
        return f"M{result.score_mate}"
    if result.score_cp is not None:
        return f"{result.score_cp / 100:+.2f}"
    return "?"


def format_pv(pv_san: list[str]) -> str:
    """Principal variation as SAN, the way a player reads it."""
    return " ".join(pv_san) if pv_san else "—"


def _row(label: str, value: object) -> str:
    return f"{label + ':':<{_LABEL_WIDTH}}{value}"


def format_analysis(
    results: list[AnalysisResult],
    *,
    engine_status: str = "Unknown",
    engine_name: str = "",
) -> str:
    """Render an analysis exactly as PHASE 8 of the spec requires.

    BEST MOVE / ALTERNATIVES / PRINCIPAL VARIATION first, then the raw
    engine data. Real engine numbers only — nothing is invented here, that
    is what PHASE 9 is for.
    """
    if not results:
        return "No results."

    best = results[0]
    blocks: list[str] = [
        f"{BEST_MOVE_HEADER}\n{best.best_move_san}\n{format_score(best)}"
    ]

    alternatives = results[1:]
    if alternatives:
        alt_lines = [ALTERNATIVES_HEADER]
        for rank, result in enumerate(alternatives, start=2):
            alt_lines.append(f"{rank}. {result.best_move_san}")
            alt_lines.append(f"   {format_score(result)}")
            alt_lines.append("")
        if alt_lines[-1] == "":
            alt_lines.pop()
        blocks.append("\n".join(alt_lines))

    blocks.append(f"{PV_HEADER}\n{format_pv(best.pv_san)}")

    stats = [
        _row("Depth", best.depth),
    ]
    if best.seldepth:
        stats.append(_row("Seldepth", best.seldepth))
    if best.nodes is not None:
        stats.append(_row("Nodes", f"{best.nodes:,}"))
    stats.extend(
        [
            _row("Analysis time", f"{best.time_seconds:.2f} s"),
            _row("Evaluation", f"{format_score(best)} (from side to move)"),
            _row(
                "Side to move",
                "white" if best.side_to_move_white else "black",
            ),
            _row("MultiPV", len(results)),
            _row("Engine status", engine_status),
        ]
    )
    if engine_name:
        stats.append(_row("Engine", engine_name))
    blocks.append(f"{ENGINE_HEADER}\n" + "\n".join(stats))

    return "\n\n".join(blocks)
