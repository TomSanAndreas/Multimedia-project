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
                current_score = {x: max([[matches[y][x], j, y] for y in matches], key=lambda z: z[0]) for x in positions}
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

        # Iets grotere (tijdelijke) array gebruiken die de resulterende
        # indices bevat (met oneven assen zodat er een middelste startpunt
        # is).
        # TODO: alle ints van waarde -1 (= unused) nemen als initiele
        # waarde, zodat er kan gecontrolleerd worden op gebruikte
        # posities
        resulting_indices = np.zeros((self.shape[0] + 1 + self.shape[0] % 2, self.shape[1] + 1 + self.shape[1] % 2), dtype=np.int)

        # rolling gebruiken om buren toe te voegen indien nodig
        # resulting_indices[0,:] = 1
        # resulting_indices = np.roll(resulting_indices, 1, axis=0)
        # print(resulting_indices)

        # Middelste index van resulting_indices is positie 0
        # (zeroe() werd gebruikt, dus moet niet meer manueel worden geplaatst)
        # resulting_indices[(self.shape[0] + 1) // 2, (self.shape[0] + 1) // 2] = 0
        
        current_piece_index = 0
        current_pos_index = [(self.shape[1] + 1) // 2, (self.shape[0] + 1) // 2]

        print(scores)
        # TODO: usable pieces updaten per stuk dat wordt toegevoegd in
        # de puzzel, zodanig dat een stuk niet meermaals wordt gebruikt
        # en eventueel hier terug in toevoegen indien een bestaand
        # stuk wordt overschreven
        usable_pieces = {i for i in range(self.shape[0] * self.shape[1])}
        # TODO: aantal rotaties in een totaal bijhouden, modulo 4 en zoveel
        # keer het totaal roteren (zou minimale rotatie moeten opleveren van
        # het totaal, wat meer kans heeft de gewenste totale rotatie te zijn)
        for i in range(self.shape[0] * self.shape[1]):
            # Of linker, of rechterbuur plaatsen (op basis van score) en eventueel
            # die buur roteren
            current_score = scores[current_piece_index]
            if i % 2:
                key, current_pos_index[0] = ("top", current_pos_index[0] - 1) if current_score["top"][0] > current_score["bottom"][0] else ("bottom", current_pos_index[0] + 1)
            else:
                key, current_pos_index[1] = ("left", current_pos_index[1] - 1) if current_score["left"][0] > current_score["right"][0] else ("right", current_pos_index[1] + 1)
            next_piece_index = current_score[key][1]
            resulting_indices[tuple(current_pos_index)] = next_piece_index
            n_rotations = current_score[key][2]
            print(current_pos_index)
            print(n_rotations)
            # stuk zelf, alsook de scores van dit stuk, roteren
            for i in range(n_rotations):
                self.pieces[next_piece_index] = self.pieces[next_piece_index].rotate()
                temp_top = scores[next_piece_index]["left"]
                scores[next_piece_index]["left"] = scores[next_piece_index]["bottom"]
                scores[next_piece_index]["bottom"] = scores[next_piece_index]["right"]
                scores[next_piece_index]["right"] = scores[next_piece_index]["top"]
                scores[next_piece_index]["top"] = temp_top
                for k in scores[next_piece_index]:
                    scores[next_piece_index][k][2] = (scores[next_piece_index][k][2] + 1) % 4
            current_piece_index = next_piece_index
            # Of boven, of onderbuur plaatsen (op basis van score) en eventueel
            # die buur roteren

            # temp
            if i == 2:
                break

        print(resulting_indices)
        # resulterende indices omzetten naar de board orientatie
        # (= de gebruikte shape eruit filteren)
        # beide kunnen max 2 zijn
        offset_h = int((resulting_indices[0, :] == 0).all()) + int((resulting_indices[1, :] == 0).all())
        offset_v = int((resulting_indices[:, 0] == 0).all()) + int((resulting_indices[:, 1] == 0).all())
        self.orientation = resulting_indices[offset_h:self.shape[0] + offset_h, offset_v:self.shape[1] + offset_v].tolist()
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
    def create_board(img, puzzle_type: int, puzzle_dims: tuple[int, int] = None):
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
                print(f"Gevonden dimensies: {n_h} x {n_v}")
                h_size, v_size = w // puzzle_dims[0], h // puzzle_dims[1]
                print(f"Gegeven dimensies: {puzzle_dims[0]} x {puzzle_dims[1]}")
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
