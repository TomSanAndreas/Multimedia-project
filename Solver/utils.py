# Algemene imports
import numpy as np
from matplotlib import pyplot as plt
from os.path import isfile
import cv2

# Eigen imports
from Solver.types import *

# algemene functies
def get_image_path(name: str) -> str:
    """
    Probeert de parameter name om te zetten naar een geldig
    pad die bestaat, geeft None terug indien het bestand
    niet bestaat.
    """
    if name[-4] != '.':
        if isfile(name + ".png"):
            return name + ".png"
        elif isfile(name + ".jpg"):
            return name + ".jpg"
    elif isfile(name):
        return name
    folder = "_".join(name.split("_")[:2]).capitalize()
    name = f"Documentatie/Puzzels/{folder}/{name}"
    if name[-4] != '.':
        if isfile(name + ".png"):
            return name + ".png"
        elif isfile(name + ".jpg"):
            return name + ".jpg"
    elif isfile(name):
        return name
    return None

def show_image(image: np.ndarray):
    """
    Geeft een plot weer met enkel de gegeven image in.
    """
    if len(image.shape) == 3:
        # BGR-beeld, dus omkeren van elementenvolgorde
        # zodat het beeld als een RGB-beeld wordt
        # weergegeven
        plt.imshow(image[:, :, ::-1])
    else:
        # Grijswaardenbeeld, gewoon weergeven
        plt.imshow(image, cmap = "gray")
    plt.axis("off")
    plt.show()

def calc_edge_score(edge: np.ndarray) -> float:
    """
    Berekent een score tussen 0.0 (totaal geen edge) en
    1.0 (meest optimale edge) afhankelijk van de inhoud
    van de edge zelf.
    edge heeft dimensie (L, W) waarvan
        L: de lengte van de edge is,
        W: de dikte van de pixels van de edge is
        (voorbeeld: 256 x 3 voor een 256 pixel edge met
        3 pixels)
    """
    index = 0
    line_length = 1
    max_line_length = 1
    while index + line_length < len(edge):
        while index + line_length < len(edge) and (edge[index:index + line_length + 1] == 255).any(axis = 1).all():
            line_length += 1
        index += line_length + 1
        max_line_length = max(max_line_length, line_length)
        line_length = 1
    return max_line_length / len(edge)

def get_file_info_by_name(filename: str) -> tuple[int, tuple[int, int]]:
    """
    Filename heeft de vorm tiletype_XxY_index.png
    """
    fileprops = filename.lower().split('/')[-1].split('_')
    tiledesc = fileprops[0:2]
    # moet altijd 1 cijfer zijn bij 1 cijfer
    tiledims = int(fileprops[2][2]), int(fileprops[2][0])
    if tiledesc[0] == "jigsaw":
        tiletype = 0
    elif tiledesc[0] == "tiles":
        tiletype = 3
    else:
        raise RuntimeError("Invalid filename detected! (1)")
    if tiledesc[1] == "rotated":
        tiletype += 1
    elif tiledesc[1] == "scrambled":
        tiletype += 2
    elif tiledesc[1] == "shuffled":
        tiletype += 3
    else:
        raise RuntimeError("Invalid filename detected! (2)")
    return tiletype, tiledims

def move(coord: tuple[int, int], key: str, n: int = 1, clip: bool = False, min: tuple[int, int] = None, max: tuple[int, int] = None) -> tuple[int, int]:
    """
    Verplaatst een coordinaat x,y volgens key n keer,
    key zit in ["left", "right", "top", "bottom"]
    """
    y, x = coord
    if key == "right":
        x += n
        x = max[1] if clip and x > max[1] else x
    elif key == "left":
        x -= n
        x = min[1] if clip and x < min[1] else x
    elif key == "bottom":
        y += n
        y = max[0] if clip and y > max[0] else y
    elif key == "top":
        y -= n
        y = min[0] if clip and y < min[0] else y
    else:
        raise RuntimeError(f"Invalid key: {key}")
    return y, x