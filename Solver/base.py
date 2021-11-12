# Python imports
import threading

# Eigen imports
from Solver.piece import PuzzlePiece
from Solver.types import Types

class Solver(threading.Thread):
    def __init__(self, source_image, puzzle_type: int = Types.GUESS):
        super(Solver, self).__init__()
        self.src = source_image
        self.puzzle_type = puzzle_type
    def solve(self):
        self.start()
    def run(self):
        if self.puzzle_type == Types.TILES_ROTATED:
            self.solve_tiles_rotated()
        else:
            print("Not yet supported!")
    def solve_tiles_rotated(self):
        pass
    def solve_tiles_scrambled(self):
        pass
    def solve_tiles_shuffled(self):
        pass
    def solve_jigsaw_rotated(self):
        pass
    def solve_jigsaw_scrambled(self):
        pass
    def solve_jigsaw_shuffled(self):
        pass