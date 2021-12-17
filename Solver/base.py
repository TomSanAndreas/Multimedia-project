# Python imports
import threading

# Eigen imports
from Solver.types import Types
from Solver.board import Board
from Solver.utils import *

class Solver(threading.Thread):
    def __init__(self, source_image, puzzle_properties: tuple[int, tuple[int, int]] = None):
        super(Solver, self).__init__()
        self.puzzle_type = puzzle_properties[0]
        self.board = Board.create_board(source_image, self.puzzle_type, puzzle_properties[1])
        self.board.show()
        self.solved = False
    def solve(self):
        self.is_ready = False
        self.start()
    def run(self):
        self.solved = self.board.solve()
        self.is_ready = True
    def show(self):
        result = self.board.create_image()
        show_image(result)
        self.board.show()
