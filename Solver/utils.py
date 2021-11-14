# Algemene imports
import numpy as np
from matplotlib import pyplot as plt
from os.path import isfile
import cv2

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