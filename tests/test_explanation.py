"""Phase 9 tests: the explanation must be objective and must invent nothing.

Every test here is pure python-chess against a hardcoded FEN: no Stockfish,
no Qt, no YOLO.  The whole file runs in milliseconds.
"""

from __future__ import annotations

import chess

from app.chess.explanation import (
    DISCLAIMER,
    EXPLANATION_HEADER,
    explain_fen,
    explain_position,
    format_explanation,
)

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Words that would make an explanation subjective rather than factual.
FORBIDDEN = (
    "good",
    "strong",
    "weak",
    "threaten",
    "controls",
    "prepares",
    "improves",
    "blunder",
    "mistake",
    "brilliant",
    "idea",
    "advantage",
    "punish",
)


def block(fen: str, uci: str | None = None) -> str:
    return explain_fen(fen, best_move_uci=uci)


class TestDisclaimer:
    def test_header_is_the_phase_nine_one(self) -> None:
        assert block(STARTPOS).splitlines()[0] == EXPLANATION_HEADER

    def test_it_states_that_stockfish_only_brings_move_and_score(self) -> None:
        assert DISCLAIMER in block(STARTPOS)
        assert "Stockfish supplies only the move and the evaluation." in DISCLAIMER

    def test_empty_input_renders_nothing(self) -> None:
        assert format_explanation([]) == ""


class TestMaterial:
    def test_startpos_is_equal(self) -> None:
        assert "Material is equal." in block(STARTPOS)

    def test_a_lead_names_the_piece_that_is_surplus(self) -> None:
        text = block("7k/8/8/8/8/8/8/K6Q w - - 0 1")
        assert "White is ahead by 9 points (queen)." in text

    def test_a_single_point_is_singular(self) -> None:
        text = block("6k1/8/8/8/8/8/8/K6P w - - 0 1")
        assert "White is ahead by 1 point (pawn)." in text


class TestCheck:
    def test_check_is_reported_for_the_side_to_move(self) -> None:
        text = block("7k/6Q1/8/8/8/8/8/K7 b - - 0 1")
        assert "Black king is in check." in text

    def test_quiet_positions_say_nothing_about_check(self) -> None:
        assert "check." not in block(STARTPOS).replace("gives check", "")


class TestCastling:
    def test_both_sides_may_still_castle(self) -> None:
        text = block(STARTPOS)
        assert "White may castle on both sides." in text
        assert "Black may castle on both sides." in text

    def test_a_side_that_lost_its_rights_is_not_offered_castling(self) -> None:
        text = block("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        assert "White may castle kingside." in text
        assert "Black may castle" not in text
        assert "White may castle on both sides." not in text

    def test_nobody_may_castle(self) -> None:
        assert "Neither side has castling rights." in block("7k/8/8/8/8/8/8/K7 w - - 0 1")


class TestBestMove:
    def test_a_capture_says_what_it_takes(self) -> None:
        text = block("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", uci="e4d5")
        assert "The best move exd5 captures a pawn." in text

    def test_a_check_is_reported(self) -> None:
        text = block("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", uci="a1a8")
        assert "The best move Ra8+ gives check." in text

    def test_mate_is_distinguished_from_check(self) -> None:
        text = block("7k/8/6K1/8/8/8/8/7Q w - - 0 1", uci="h1h7")
        assert "The best move Qh7# gives mate." in text

    def test_a_quiet_move_says_what_it_does_not_do(self) -> None:
        text = block(STARTPOS, uci="e2e4")
        assert "The best move e4 neither captures nor gives check." in text

    def test_castling_move_is_called_out(self) -> None:
        # The starting position blocks e1g1, so use an open back rank.
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        text = block(fen, uci="e1g1")
        assert "The best move O-O castles kingside." in text

    def test_a_blocked_king_may_not_claim_to_castle(self) -> None:
        text = block(STARTPOS, uci="e1g1")
        assert "castles" not in text
        assert "The best move" not in text

    def test_no_move_just_leaves_the_position_facts(self) -> None:
        text = block(STARTPOS)
        assert "The best move" not in text
        assert "Material is equal." in text


class TestPositionFacts:
    def test_development_is_counted_off_the_back_rank(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
        text = block(fen)
        assert "White has 3 of 4 knights and bishops on the back rank." in text
        assert "Black has 4 of 4 knights and bishops on the back rank." in text

    def test_an_undeveloped_position_says_nothing_about_development(self) -> None:
        assert "back rank" not in block(STARTPOS)

    def test_an_empty_centre_is_stated(self) -> None:
        assert "None of d4, e4, d5 or e5 is occupied." in block(STARTPOS)

    def test_occupied_centre_squares_are_named(self) -> None:
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        text = block(fen)
        assert "White pawn on e4" in text
        assert "Black pawn on e5" in text


class TestItNeverInvents:
    def test_no_output_contains_a_subjective_judgement(self) -> None:
        positions = (
            STARTPOS,
            "7k/6Q1/8/8/8/8/8/K7 b - - 0 1",
            "rnbq1rk1/ppp2ppp/3p4/3B4/4P3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        )
        for fen in positions:
            text = explain_fen(fen).lower()
            for word in FORBIDDEN:
                assert word not in text, f"{word!r} appeared in the explanation"

    def test_every_line_is_derived_from_the_fen(self) -> None:
        # The same FEN must always produce the same explanation: nothing is
        # sampled, guessed or looked up.
        assert explain_fen(STARTPOS) == explain_fen(STARTPOS)

    def test_explain_position_ignores_an_illegal_move(self) -> None:
        board = chess.Board(STARTPOS)
        lines = explain_position(board, best_move=chess.Move.from_uci("e2e5"))
        assert not any(line.startswith("The best move") for line in lines)

    def test_a_malformed_fen_renders_nothing(self) -> None:
        assert explain_fen("not a fen") == ""
