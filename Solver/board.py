# Eigen imports
from Solver.piece import PuzzlePiece
from Solver.types import Types

class Board:
    def __init__(self, pieces):
        self.pieces = pieces

    @staticmethod
    def create_board(img, puzzle_type: int) -> Board:
        # Create_Board gaat het type puzzel niet raden, dit moet
        # appart gebeuren!
        assert(puzzle_type != Types.GUESS)
        if Types.is_tiled(puzzle_type):
            # Pieces zijn van type Tiled
            # Kijken indien het om een rotated board gaat of niet
            # TODO
            pass
        else:
            # pieces zijn van type Jigsaw
            # Kijken indien het om een rotated board gaat of niet
            # TODO
            pass
        return Board([])