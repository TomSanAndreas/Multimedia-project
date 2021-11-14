# Algemene imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Eigen imports
from Solver.piece import *
from Solver.types import Types
from Solver.utils import *

class Board:
    def __init__(self, pieces, shape):
        self.pieces = pieces
        self.shape = shape
        self.orientation = [[y * shape[0] + x for x in range(shape[0])] for y in range(shape[1])]

    def show(self) -> None:
        fig = plt.figure()
        index = 0
        for row in self.orientation:
            for piece in row:
                index += 1
                ax = fig.add_subplot(self.shape[1], self.shape[0], index)
                ax.imshow(self.pieces[piece].img[:,:,::-1])
                ax.axis("off")
        plt.show()

    def solve(self) -> bool:
        """
        Probeert het bord op te lossen
        Geeft True indien succesvol
        """


        # # Kijken hoeveel buren er verwacht worden a.d.h.v. de afmeting
        # n_horizontal = 1 + int(self.shape[0] > 2)
        # n_vertical = 1 + int(self.shape[1] > 2)


        # zoeken van een hoekstuk door van elk stuk
        # de vergelijking te maken met elk ander stuk
        # en het stuk met de kleinste overeenkomsten
        # voor 2 zijden (boven-onder/links-rechts)
        # als hoekstuk te veronderstellen
        scores = []
        positions = ["left", "right", "top", "bottom"]
        for i in range(len(self.pieces)):
            scores.append({j: [0, None, None] for j in positions})
            for j in set(range(len(self.pieces))) - {i}:
                matches = self.pieces[i].match_all(self.pieces[j])
                # enkel maxima overhouden, zodat er makkelijker gekeken kan worden naar het aantal buren
                current_score = {x: max([(matches[y][x], j, y) for y in matches], key=lambda z: z[0]) for x in positions}
                scores[-1] = {x: max((scores[y][x], current_score[x]), key=lambda x: x[0]) for y in range(len(scores)) for x in positions}
        # scores is nu een lijst die voor elk stuk de beste scores
        # bevat, de index van het stuk die daar het best past en
        # de rotatie die dat stuk moet aannemen voor deze best-
        # passende score
        # bijvoorbeeld:
        # ---
        # [
        #     {
        #         'left': (0.11372549019607843, 3, 0),
        #         'right': (0.32941176470588235, 3, 0),
        #         'top': (0.1411764705882353, 3, 2),
        #         'bottom': (0.32941176470588235, 2, 2)
        #     }, {
        #         ... (dit voor elke tegel)
        #     }
        # ]
        # ---
        # Het eerste stuk hierboven heeft zo duidelijk een rechter- en
        # onderbuur, aangezien daar een 33% match gevonden is met stuk
        # 3 respectievelijk 2, waarvan stuk 3 niet meer moet gedraaid
        # worden, en stuk 2 180° moet gedraaid worden.


        # if n_horizontal == 1:
        #     # er wordt maar 1 horizontale buur gezocht voor eender welk
        #     # stuk, dus op basis van de scores kan deze eenvoudig gekozen
        #     # worden (zolang er geen conflichten voorkomen)
        #     pass
        # else:
        #     # eerst een hoekpunt zoeken
        #     # TODO
        #     raise NotImplementedError()
        # if n_vertical == 1:
        #     # idem voor verticale buren
        #     pass
        # else:
        #     # eerst een hoekpunt zoeken
        #     # TODO
        #     raise NotImplementedError()

        # Alle mogelijke linken maken, waarbij enkel de sterkste overblijven
        for i, score in enumerate(scores):
            for j, pos in enumerate(positions):
                self.pieces[i].add(self.pieces[score[pos][1]], score[pos][0], j)
        # Zwakste linken wegfilteren zodanig dat er voldoende stukken zijn
        # die als randen/hoeken kunnen worden herkend
        # TODO
        for piece in self.pieces:
            print([r[1] != None for r in piece.relations])
        return True
    
    def create_image(self) -> np.ndarray:
        """
        Creeert een afbeelding met de stukken op de
        ingestelde posities en geeft deze terug
        """
        result = np.concatenate([self.pieces[i].img for i in self.orientation[0]], axis=1)
        for row in self.orientation[1:]:
            result_row = np.concatenate([self.pieces[i].img for i in row], axis=1)
            result = np.concatenate((result, result_row))
        return result

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
                threshold1 = 400
                threshold2 = 400
                edges = cv2.Canny(img, threshold1, threshold2)
                h, w = img.shape[0:2]
                while np.count_nonzero(edges) / (h * w) < .05:
                    threshold1 -= 10
                    edges = cv2.Canny(img, threshold1, threshold2)
                while np.count_nonzero(edges) / (h * w) > .1:
                    threshold2 += 10
                    edges = cv2.Canny(img, threshold1, threshold2)
                # detecteren aantal horizontale stukken
                # er kunnen 2 tot 5 stukken horizontaal liggen
                max_line_score, detected_edge = 0, 0
                for i in range(5, 1, -1):
                    current_i_score = 0
                    for j in range(1, i):
                        px = int(w / i) * j
                        current_i_score += calc_edge_score(edges[:, px - 1:px + 2])
                    current_i_score /= (i - 1)
                    if current_i_score > max_line_score:
                        max_line_score = current_i_score
                        detected_edge = i
                n_h = detected_edge
                # idem voor het detecteren van het aantal verticale stukken
                max_line_score, detected_edge = 0, 0
                for i in range(5, 1, -1):
                    current_i_score = 0
                    for j in range(1, i):
                        px = int(h / i) * j
                        current_i_score += calc_edge_score(edges[px - 1:px + 2, :].T)
                    current_i_score /= (i - 1)
                    if current_i_score > max_line_score:
                        max_line_score = current_i_score
                        detected_edge = i
                n_v = detected_edge
                # met n_h en n_v de img opsplitsen in stukken en teruggeven
                h_size = int(w / n_h)
                v_size = int(h / n_v)
                return Board([TiledPiece(img[v_size * y:v_size * (y + 1), h_size * x:h_size * (x + 1)]) for y in range(n_v) for x in range(n_h)], (n_h, n_v))
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

"""
Oude code (backup)
--- Detecteren aantal stukken van een tiled (en niet scrambled) puzzel ---

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

"""