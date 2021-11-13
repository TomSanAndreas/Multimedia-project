# Algemene imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Eigen imports
from Solver.piece import PuzzlePiece
from Solver.types import Types
from Solver.utils import *

class Board:
    def __init__(self, pieces, shape):
        self.pieces = pieces
        self.shape = shape

    def show(self) -> None:
        fig = plt.figure()
        index = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                index += 1
                ax = fig.add_subplot(self.shape[0], self.shape[1], index)
                ax.imshow(self.pieces[j * self.shape[0] + i][:,:,::-1])
                ax.axis("off")
        plt.show()

    @staticmethod
    def create_board(img, puzzle_type: int):
        # Create_Board gaat het type puzzel niet raden, dit moet
        # appart gebeuren!
        assert(puzzle_type != Types.GUESS)
        if Types.is_tiled(puzzle_type):
            # Pieces zijn van type Tiled
            # Kijken indien het om een scrambled board gaat of niet,
            # aangezien de aanpak voor het detecteren van stukken
            # dan anders verloopt
            if puzzle_type != Types.TILES_SCRAMBLED:
                # De stukken liggen naast elkaar op de gegeven foto,
                # en kunnen dus, eenmaal het aantal stukken gevonden
                # is, eenvoudig gevonden worden (aangezien elk stuk
                # even groot is)
                edges = cv2.Canny(img, 250, 400)
                # threshold dynamisch regelen tot het aantal gevonden lines groot genoeg is
                lines = [[], []]
                threshold = 2000
                while (len(lines[0]) < 1 or len(lines[1]) < 1) and threshold > 50:
                    threshold -= 45
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold, None, 0, 0)
                    lines = [] if lines is None else lines
                    # enkel lijnen met een theta = 0 en 90° zijn relevant,
                    # dus verkeerde wegfilteren
                    lines = [[line[0] for line in lines if 0 <= line[0][1] <= 0.001], [line[0] for line in lines if np.pi / 2 - 0.0005 <= line[0][1] <= np.pi / 2 + 0.0005]]
                # de lijnen in lines kunnen gebruikt worden om te weten hoeveel stukken er
                # zijn (n x m)
                # horizontaal aantal stukken bepalen
                n_h = 10
                for line in lines[0]:
                    for i in range(-1, 2):
                        n_horizontal = 1
                        ratio = (line[0] + i) / img.shape[1]
                        while n_horizontal < 10 and ratio * n_horizontal != int(ratio * n_horizontal):
                            n_horizontal += 1
                        if n_h > n_horizontal:
                            n_h = n_horizontal
                if n_h == 10:
                    raise RuntimeError("Aantal horizontale stukken werd niet correct gedetecteerd!")
                # verticaal aantal stukken bepalen
                n_v = 10
                for line in lines[1]:
                    for i in range(-1, 2):
                        n_vertical = 1
                        ratio = (line[0] + i) / img.shape[0]
                        while n_vertical < 10 and ratio * n_vertical != int(ratio * n_vertical):
                            n_vertical += 1
                        if n_v > n_vertical:
                            n_v = n_vertical
                if n_v == 10:
                    raise RuntimeError("Aantal verticale stukken werd niet correct gedetecteerd!")
                # met n_h en n_v de img opsplitsen in stukken en teruggeven
                h_size = int(img.shape[1] / n_h)
                v_size = int(img.shape[0] / n_v)
                return Board([img[h_size * x:h_size * (x + 1), v_size * y:v_size * (y + 1)] for x in range(n_h) for y in range(n_v)], (n_h, n_v))
            else:
                # De stukken liggen verspreid op het beeld, met
                # zwartruimtes tussen.
                # TODO
                raise NotImplementedError()
        else:
            # pieces zijn van type Jigsaw
            # Kijken indien het om een rotated board gaat of niet
            # TODO
            raise NotImplementedError()
        return Board([], (0, 0))