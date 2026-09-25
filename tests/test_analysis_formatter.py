import chess
import chess.engine

from app.engine.stockfish_engine import AnalysisResult, EngineManager
from app.gui.analysis_formatter import (
    format_analysis,
    format_pv,
    format_score,
)

_RESULT_DEFAULTS: dict = dict(
    multipv=1,
    score_cp=84,
    score_mate=None,
    depth=22,
    seldepth=30,
    nodes=1_234_567,
    time_seconds=5.5,
    pv_uci=["g1f3"],
    pv_san=["Nf3"],
    best_move_san="Nf3",
    is_mate=False,
    side_to_move_white=True,
)


def make_result(**overrides) -> AnalysisResult:
    data = {**_RESULT_DEFAULTS, **overrides}
    return AnalysisResult(**data)


def build_result(fen: str, pov_score: chess.engine.Score) -> AnalysisResult:
    """Exercise EngineManager._build_result without starting Stockfish."""
    board = chess.Board(fen)
    info = {
        "score": chess.engine.PovScore(pov_score, chess.WHITE),
        "pv": [board.parse_san("Nf3")]
        if board.turn == chess.WHITE
        else [chess.Move.from_uci("e7e5")],
        "depth": 22,
        "seldepth": 30,
        "nodes": 1_234_567,
        "time": 5.5,
        "multipv": 1,
    }
    return EngineManager()._build_result(board, info)


class TestFormatScore:
    def test_centipawn(self) -> None:
        assert format_score(make_result(score_cp=84)) == "+0.84"

    def test_negative_centipawn(self) -> None:
        assert format_score(make_result(score_cp=-50)) == "-0.50"

    def test_zero_centipawns(self) -> None:
        assert format_score(make_result(score_cp=0)) == "+0.00"

    def test_mate_positive(self) -> None:
        assert format_score(
            make_result(score_cp=None, score_mate=5, is_mate=True)
        ) == "M5"

    def test_mate_negative(self) -> None:
        assert format_score(
            make_result(score_cp=None, score_mate=-3, is_mate=True)
        ) == "M-3"

    def test_missing_score(self) -> None:
        assert format_score(
            make_result(score_cp=None, score_mate=None, is_mate=False)
        ) == "?"


class TestFormatPv:
    def test_joins_san_moves(self) -> None:
        assert format_pv(["Nf3", "e5", "Nc3", "Nc6", "Bb5"]) == (
            "Nf3 e5 Nc3 Nc6 Bb5"
        )

    def test_empty_variation(self) -> None:
        assert format_pv([]) == "—"


class TestFormatAnalysis:
    def test_empty_results(self) -> None:
        assert format_analysis([]) == "No results."

    def test_sections_follow_the_spec_order(self) -> None:
        results = [make_result(), make_result(multipv=2, best_move_san="d4")]
        text = format_analysis(results)

        order = [
            "BEST MOVE",
            "ALTERNATIVES",
            "PRINCIPAL VARIATION",
            "--- ENGINE DATA ---",
        ]
        positions = [text.index(header) for header in order]
        assert positions == sorted(positions)

    def test_best_move_block(self) -> None:
        text = format_analysis([make_result()])
        block = text.split("\n\n")[0]
        assert block == "BEST MOVE\nNf3\n+0.84"

    def test_alternatives_are_ranked_from_two(self) -> None:
        results = [
            make_result(),
            make_result(multipv=2, best_move_san="d4", score_cp=72),
            make_result(multipv=3, best_move_san="c4", score_cp=61),
        ]
        text = format_analysis(results)

        assert "ALTERNATIVES" in text
        assert "2. d4\n   +0.72" in text
        assert "3. c4\n   +0.61" in text

    def test_single_variation_has_no_alternatives_block(self) -> None:
        text = format_analysis([make_result()])
        assert "ALTERNATIVES" not in text

    def test_principal_variation_in_san(self) -> None:
        text = format_analysis(
            [make_result(pv_san=["Nf3", "e5", "Nc3"], best_move_san="Nf3")]
        )
        assert "PRINCIPAL VARIATION\nNf3 e5 Nc3" in text

    def test_every_required_stat_is_displayed(self) -> None:
        text = format_analysis(
            [make_result()], engine_status="Idle", engine_name="Stockfish 17.1"
        )
        for label in (
            "Depth:",
            "Nodes:",
            "Analysis time:",
            "Evaluation:",
            "Side to move:",
            "MultiPV:",
            "Engine status:",
            "Engine:",
        ):
            assert label in text, f"missing {label}"

    def test_engine_status_and_name(self) -> None:
        text = format_analysis(
            [make_result()], engine_status="Idle", engine_name="Stockfish 17.1"
        )
        assert "Engine status: Idle" in text
        assert "Engine:        Stockfish 17.1" in text

    def test_multi_pv_count(self) -> None:
        results = [make_result(), make_result(multipv=2), make_result(multipv=3)]
        assert "MultiPV:       3" in format_analysis(results)

    def test_evaluation_is_read_from_the_side_to_move(self) -> None:
        white = format_analysis(
            [make_result(score_cp=84, side_to_move_white=True)]
        )
        black = format_analysis(
            [make_result(score_cp=-84, side_to_move_white=False)]
        )

        assert "Evaluation:    +0.84 (from side to move)" in white
        assert "Side to move:  white" in white
        assert "Evaluation:    -0.84 (from side to move)" in black
        assert "Side to move:  black" in black

    def test_nodes_and_time_are_rendered(self) -> None:
        text = format_analysis(
            [make_result(nodes=1_234_567, time_seconds=5.5)]
        )
        assert "Nodes:         1,234,567" in text
        assert "Analysis time: 5.50 s" in text

    def test_no_trailing_whitespace(self) -> None:
        text = format_analysis(
            [make_result(), make_result(multipv=2, best_move_san="d4")]
        )
        for line in text.split("\n"):
            assert line == line.rstrip()


class TestEvaluationPointOfView:
    """The score must come back from the side to move, not always White."""

    STARTPOS = chess.STARTING_FEN
    BLACK_TO_MOVE = (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )

    def test_white_to_move_keeps_a_positive_score(self) -> None:
        result = build_result(self.STARTPOS, chess.engine.Cp(84))
        assert result.score_cp == 84
        assert result.side_to_move_white is True

    def test_black_to_move_flips_the_sign(self) -> None:
        result = build_result(self.BLACK_TO_MOVE, chess.engine.Cp(84))
        assert result.score_cp == -84
        assert result.side_to_move_white is False

    def test_mate_score_flips_with_the_side_to_move(self) -> None:
        white = build_result(self.STARTPOS, chess.engine.Mate(5))
        black = build_result(self.BLACK_TO_MOVE, chess.engine.Mate(5))
        assert white.score_mate == 5
        assert black.score_mate == -5

    def test_mate_flag_survives_the_conversion(self) -> None:
        result = build_result(self.STARTPOS, chess.engine.Mate(5))
        assert result.is_mate is True
        assert result.score_cp is None

    def test_pv_is_still_san(self) -> None:
        result = build_result(self.STARTPOS, chess.engine.Cp(84))
        assert result.pv_san == ["Nf3"]
        assert result.best_move_san == "Nf3"
