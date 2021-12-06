# Algemene imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Eigen imports
from Solver.piece import *
from Solver.types import Types
from Solver.utils import *

class Board:
    def __init__(self, pieces: list['Piece'], shape: tuple[int, int]):
        # alle gebruikte stukken van dit bord bijgehouden in een lijst
        self.pieces = pieces
        # aantal stukken horizontaal, verticaal
        self.shape = shape
        # lijst van lijsten die alle indices van het (opgeloste) bord bevatten
        self.orientation = [[y * shape[0] + x for x in range(shape[0])] for y in range(shape[1])]
        # initiele waarde, gebruikt bij het oplossen van de puzzel
        self.threshold = 5

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
        print("\033[1mSolving...\033[0m")
        attempt_index = 1
        while self.threshold < 15:
            print(f"\033[4mAttempt {attempt_index:2}/15\033[0m: Active" + (" " * 20))
            success, result = self.try_solve()
            if success:
                self.orientation = result
                # go a line back and update the header, with extra newline so status remains visible
                print(f"\033[1A\033[4mAttempt {attempt_index:2}/15\033[0m: \033[92mSucces!\033[0m" + (" " * 20), end='\n\n')
                return True
            # niet succesvol, alle orientaties terugzetten, threshold
            # verlagen en opnieuw proberen
            self.pieces = [p.rotate_to(0) for p in self.pieces]
            self.threshold += 1
            # go a line back and update the status
            print(f"\033[1A\033[4mAttempt {attempt_index:2}/15\033[0m: \033[91mFailed!\033[0m" + (" " * 20))
            attempt_index += 1

        # threshold resetten en naar beneden toe werken
        self.threshold = 5
        while self.threshold > 0:
            print(f"\033[4mAttempt {attempt_index:2}/15\033[0m: Active" + (" " * 20))
            success, result = self.try_solve()
            if success:
                self.orientation = result
                # go a line back and update the header, with extra newline so status remains visible
                print(f"\033[1A\033[4mAttempt {attempt_index:2}/15\033[0m: \033[92mSucces!\033[0m" + (" " * 20), end='\n\n')
                return True
            # niet succesvol, alle orientaties terugzetten, threshold
            # verlagen en opnieuw proberen
            self.pieces = [p.rotate_to(0) for p in self.pieces]
            self.threshold -= 1
            # go a line back & update the header
            print(f"\033[1A\033[4mAttempt {attempt_index:2}/15\033[0m: \033[91mFailed!\033[0m" + (" " * 20))
            attempt_index += 1

        # puzzel werd niet correct opgelost, toch een poging instellen
        self.orientation = result
        return False

    def try_solve(self) -> tuple[bool, list[list]]:
        """
        "Private" functie, probeert de puzzel op te lossen met de
        huidige parameters, geeft false terug indien niet succesvol,
        alsook de (beste) oplossing
        """
        # array die alle mogelijke plaatsen bevat, groot genoeg om de puzzel
        # in eender welke richting op te lossen
        indices = np.zeros((self.shape[1] * 2 - 1, self.shape[0] * 2 - 1), dtype=np.int8) - 1
        min_bounds = [0, 0]
        max_bounds = [self.shape[1] * 2 - 2, self.shape[0] * 2 - 2]
        # eerste stuk (index 0) plaatsen
        indices[self.shape[1] - 1, self.shape[0] - 1] = 0
        # bijhouden welke stukken gebruikt zijn & waar voor lookup
        used_indices = {0: (self.shape[1] - 1, self.shape[0] - 1)}
        unused_indices = {i for i in range(len(self.pieces))} - {0}
        # de volgende stukken een voor een plaatsen op basis van de beste match
        while len(unused_indices) - 1:
            # scores herberekenen volgens de stukken die al gelegd geweest zijn
            # en enkel rekening houdende met deze die nog gelegd moeten worden
            scores = self.create_matching_struct(used_indices, unused_indices)
            # zoeken naar een stuk die geplaatst kan worden op een ongebruikte plaats
            skip = 0
            while skip < min(len(used_indices), len(unused_indices)):
                p_index, key, n_index, r_index = self.get_next_match_pair(scores, skip)
                # p_index zit in used_indices, en de positie van dit stuk moet opgeschoven
                # worden volgens key, zodat de nieuwe positie kan ingevuld worden met het
                # nieuwe stuk
                # print(f"Current result:\n{indices}\nCurrent index: {p_index}, current key: {key}, next index: {n_index}")
                n_pos = move(used_indices[p_index], key, clip = True, min = min_bounds, max = max_bounds)
                if n_pos not in used_indices.values():
                    # geldige positie, het volgende stuk is gekend
                    break
                # indien deze positie al gebruikt geweest is, of deze positie is niet geldig
                # wegens de totale lengte van het bord, wordt deze mogelijkheid overgeslagen
                skip += 1
            if skip == min(len(used_indices), len(unused_indices)):
                # huidige tussenoplossing instellen
                return False, self.clean_up_result(indices, self.shape)
            # stuk plaatsen in puzzel en positie onthouden
            indices[n_pos] = n_index
            used_indices[n_index] = n_pos
            # min_bounds en max_bounds updaten indien nodig
            min_bounds[0] = max(min_bounds[0], n_pos[0] - self.shape[1] + 1)
            min_bounds[1] = max(min_bounds[1], n_pos[1] - self.shape[0] + 1)
            max_bounds[0] = min(max_bounds[0], n_pos[0] + self.shape[1] - 1)
            max_bounds[1] = min(max_bounds[1], n_pos[1] + self.shape[0] - 1)
            # geplaatste stuk draaien
            self.pieces[n_index] = self.pieces[n_index].rotate_to(r_index)
            # stuk niet meer als unused markeren
            unused_indices -= {n_index}
            # status visualiseren
            print("[" + ("=" * len(used_indices)) + (" " * (len(unused_indices) - 1)) + "] " + str(int((len(used_indices) + 1) / (self.shape[0] * self.shape[1]) * 100)) + "%")
            temp_result = self.clean_up_result(indices, self.shape)
            for row in temp_result:
                print(" " * 30)
                for el in row:
                    print(f" {el:2} " if el != -1 else " ** ", end="")
            print(f"\033[{self.shape[0] + 2}A")
        # laatste stuk plaatsen is vrij eenvoudig: matchen met een
        # geplaatst stuk en de orientatie zo bepalen tijdens het plaatsen
        # eerst oplossing vereenvoudigen zodat result nu alle
        # elementen en 1x een -1 bevat
        result = self.clean_up_result(indices, self.shape)
        # zoeken naar de plaats met de -1 in
        found = False
        for y, row in enumerate(result):
            if found:
                break
            for x, el in enumerate(row):
                if el == -1:
                    last_pos = y, x
                    found = True
                    break
        # stuk op last_pos plaatsen
        last_piece = unused_indices.pop()
        result[last_pos[0]][last_pos[1]] = last_piece
        # een stuk naast last_pos gebruiken om te matchen met het enige
        # ongebruikte stuk om te weten hoe vaak deze gedraaid moet worden
        # horizontale buur gebruiken
        ref_piece = result[last_pos[0]][last_pos[1] + 1] if last_pos[1] == 0 else result[last_pos[0]][last_pos[1] - 1]
        key = "left" if last_pos[1] == 0 else "right"
        score = self.pieces[ref_piece].match_all(self.pieces[last_piece], self.threshold)
        n_rotations = max(score.keys(), key=lambda i: score[i][key])
        self.pieces[last_piece] = self.pieces[last_piece].rotate_to(n_rotations)
        print("\033[1A")
        return True, result
    
    @staticmethod
    def clean_up_result(arr: np.ndarray, shape: tuple[int, int]) -> list:
        """
        "Private" helperfunctie, gebruikt om een te grote
        numpy array om te zetten naar een compactere lijst
        van lijsten
        """
        offset_h = 0
        while (arr[offset_h, :] == -1).all():
            offset_h += 1
        offset_v = 0
        while (arr[:, offset_v] == -1).all():
            offset_v += 1
        return arr[offset_h:shape[1] + offset_h, offset_v:shape[0] + offset_v].tolist()

    def create_matching_struct(self, matched_indices: set[int], to_be_matched_indices: set[int]) -> list[dict]:
        """
        Creeert een struct met vorm
        [
            {
                'left': [beste_match_score, beste_match_score_index, aantal_keren_index_moet_geroteerd_worden],
                'right': [..., ..., ...],
                'top': ...,
                'bottom': ...
            }, {
                ... (dit voor elke tegel van de matched_indices, waarbij beste_match_score_index komt uit de
                    to_be_matched_indices set)
            }
        ]
        Stukken die niet in de bruikbare set vermeld staan, krijgen een match
        van de vorm [0, None, None]
        """
        scores = []
        positions = ["left", "right", "top", "bottom"]
        for i in range(len(self.pieces)):
            scores.append({j: [0, None, None] for j in positions})
            if i in matched_indices:
                for j in to_be_matched_indices - {i}:
                    matches = self.pieces[i].match_all(self.pieces[j], self.threshold)
                    # enkel maxima overhouden, zodat er makkelijker gekeken kan worden naar het aantal buren
                    current_score = {x: max([[matches[y][x], j, y] for y in matches], key=lambda z: z[0]) for x in positions}
                    scores[-1] = {x: max((scores[y][x], current_score[x]), key=lambda x: x[0]) for y in range(len(scores)) for x in positions}
        return scores
    
    @staticmethod
    def get_next_match_pair(score_struct: list[dict], skip: int = 0) -> tuple[int, str, int, int]:
        """
        Interpreteert een matching struct en geeft drie getallen en een key terug,
        waarvan de eerste de index is van het stuk met de hoogste score, de tweede 
        het stuk waarmee deze hoogste score kan bekomen worden en de derde het aantal
        rotaties dat er nodig zijn (0, 90°, 180° & 270°)
        """
        index = sorted(range(len(score_struct)), key=lambda i: score_struct[i][max(score_struct[i], key=score_struct[i].get)][0], reverse=True)[skip]
        key = max(score_struct[index], key=score_struct[index].get)
        return index, key, score_struct[index][key][1], score_struct[index][key][2]

    def create_image(self) -> np.ndarray:
        """
        Creeert een afbeelding met de stukken op de
        ingestelde posities en geeft deze terug
        """
        # Indien er child board zijn, moeten deze images eerst gemaakt worden, en worden
        # deze gematched
        result = np.concatenate([self.pieces[i].img for i in self.orientation[0]], axis=1)
        black = np.zeros(self.pieces[0].img.shape, dtype=np.uint8)
        for row in self.orientation[1:]:
            result_row = np.concatenate([self.pieces[i].img if i != -1 else black for i in row], axis=1)
            result = np.concatenate((result, result_row))
        return result

    @staticmethod
    def create_board(img, puzzle_type: int, puzzle_dims: tuple[int, int] = None):
        # Create_Board gaat het type puzzel niet raden, dit moet
        # apart gebeuren!
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
                print(f"Gevonden dimensies: {n_h} x {n_v}, gegeven dimensies: {puzzle_dims[0]} x {puzzle_dims[1]}")
                h_size, v_size = w // puzzle_dims[0], h // puzzle_dims[1]
                n_h, n_v = puzzle_dims[0], puzzle_dims[1]
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
