#Alternatieve main zonder audio
# Algemene imports
import cv2
from time import sleep

# Eigen imports
from Solver.base import Solver
from Solver.types import Types
from Solver.utils import *

# main functie
if __name__ == "__main__":
    # bestandsnaam opvragen
    filename = input("Geef de naam van het bestand op (standaard: \"tiles_rotated_2x2_00\"): ")
    if len(filename) == 0:
        filename = "tiles_rotated_2x2_00"
    # bestandsnaam proberen omzetten naar een geldig pad
    filename = get_image_path(filename)
    if filename is None:
        print("Bestand werd niet gevonden!")
    else:
        type_puzzle = get_file_info_by_name(filename)
        src_image = cv2.imread(filename)
        solver = Solver(src_image, type_puzzle)
        solver.solve()

        solver.show()
