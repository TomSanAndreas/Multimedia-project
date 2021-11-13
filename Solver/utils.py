# Algemene imports
import numpy as np
from matplotlib import pyplot as plt
from os.path import isfile
import cv2

# algemene constantes
laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

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

def create_edge_image(image: np.ndarray):
    """
    Creeert een edgebeeld via een sobelmasker
    """
    return cv2.filter2D(image, -1, laplace_filter)