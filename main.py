import cv2
from Solver.base import Solver
from Solver.types import Types

if __name__ == "__main__":
    filename = input("Geef de naam van het bestand op: ")
    # type_puzzle = input("Geef het type puzzel op: ")
    # type_puzzle = Types.convert_input_to_type(type_puzzle)
    type_puzzle = Types.TILES_ROTATED
    src_image = cv2.imread(filename)
    solver = Solver(src_image, type_puzzle)
    solver.solve()
    solver.join()