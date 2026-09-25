import pytest
import chess


class TestFenValidation:
    def test_starting_position_parses(self) -> None:
        board = chess.Board(chess.STARTING_FEN)
        assert board.is_valid()
        assert board.fen() == chess.STARTING_FEN

    def test_valid_middlegame_fen(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        board = chess.Board(fen)
        assert board.is_valid()

    def test_checkmate_position(self) -> None:
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        board = chess.Board(fen)
        assert board.is_valid()

    def test_invalid_fen_string(self) -> None:
        with pytest.raises(ValueError):
            chess.Board("this is not a valid FEN")

    def test_malformed_fen_too_few_fields(self) -> None:
        with pytest.raises(ValueError):
            chess.Board("rnbqkbnr/pppppppp/8/8")

    def test_side_to_move_black(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        board = chess.Board(fen)
        assert board.is_valid()
        assert board.turn == chess.BLACK

    def test_side_to_move_white(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        assert board.is_valid()
        assert board.turn == chess.WHITE

    def test_board_copy_matches_fen(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        board = chess.Board(fen)
        board2 = chess.Board(board.fen())
        assert board.fen() == board2.fen()
