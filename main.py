# Algemene imports
import cv2
from time import sleep

# Eigen imports
from Solver.base import Solver
from Solver.types import Types
from Solver.utils import *
from Audio.base import *

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
        try:
            while not solver.is_ready:
                wait_tone = Audio(wait_notes)
                wait_tone.play(time = 0.2)
                while wait_tone.is_playing and not solver.is_ready:
                    sleep(0.2)
                if solver.is_ready:
                    wait_tone.stop()
                wait_tone.join()
        except KeyboardInterrupt:
            print("\033[2A\033[1m=> Stopsignaal ontvangen. Laatste iteratie afwerken...\033[0m\n")
            solver.force_stop()
            try:
                wait_tone.stop()
                wait_tone.join()
            except:
                # de wachttoon werd niet gestart, mag genegeerd worden
                pass
        solver.join()
        wait_tone.stop()
        end_tone = Audio(victory_notes if solver.solved else fail_notes)
        end_tone.play(time = 0.2 if solver.solved else 0.25)
        solver.show()
        end_tone.join()
