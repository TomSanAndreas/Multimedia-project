# Algemene imports
import numpy as np
from matplotlib import pyplot as plt
import cv2


class Orientation:
    LEFT_RIGHT = 0
    RIGHT_LEFT = 1
    TOP_BOTTOM = 2
    BOTTOM_TOP = 3

    REVERSE = [RIGHT_LEFT, LEFT_RIGHT, BOTTOM_TOP, TOP_BOTTOM]

class Piece:
    
    def __init__(self, img: np.ndarray, rot: int = 0, relations: 'list[tuple[float, Piece]]' = None):
        self.img = img
        self.rot = rot
        if relations is None:
            self.relations = [(0, None) for _ in range(4)]
        else:
            self.relations = relations
    
    def show(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.img[:,:,::-1])
        ax.axis("off")
        plt.show()
    
    def rotate(self) -> "Piece":
        """
        Creeert een nieuw stuk die 90° CW gedraaid is.
        """
        return Piece(cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE), (self.rot + 1) % 4)
    
    def rotate_to(self, absolute_rot: int) -> "Piece":
        """
        Draait het stuk naar een absolute rotatie (0 is zoals gegeven,
        1 is 90° gedraaid, ...)
        """
        n = (self.rot - absolute_rot) % 4 - 1
        if n == -1:
            return Piece(self.img, absolute_rot)
        code = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        return Piece(cv2.rotate(self.img, code[n]), absolute_rot)

    def can_add(self, strength: float, orientation: 'Orientation') -> bool:
        return self.relations[orientation][0] < strength
    
    def add(self, other: 'Piece', strength: float, orientation: 'Orientation') -> bool:
        """
        Probeert een relatie met een ander stuk bij te houden indien mogelijk.
        Niet mogelijk indien er al een relatie bestaat die sterker is op de
        gegeven oriëntatie, of het gegeven stuk al een relatie heeft in de tegengestelde
        richting die sterker is dan de gegeven relatie. Geeft True terug indien succesvol
        """
        reverse_orientation = Orientation.REVERSE[orientation]
        if other.can_add(strength, reverse_orientation) and self.relations[orientation][0] < strength:
            # losbreken andere relatie van huidige buur indien deze bestaat
            if self.relations[orientation][1] != None:
                self.relations[orientation][1].relations[reverse_orientation] = (0, None)
            # losbreken andere relatie indien deze bestaat van het andere element
            if other.relations[reverse_orientation][1] != None:
                other.relations[reverse_orientation][1].relations[orientation] = (0, None)
            other.relations[reverse_orientation] = (strength, self)
            self.relations[orientation] = (strength, other)
            return True
        return False

class TiledPiece(Piece):
    
    def __init__(self, img: np.ndarray, rot: int = 0, relations: 'list[tuple[float, TiledPiece]]' = None):
        super(TiledPiece, self).__init__(img, rot, relations)

    def match_all(self, other: 'TiledPiece', accuracy: int = 10, max_offset: tuple[int, int] = (-2, 2)) -> dict:
        """
        Geeft scores in de vorm van een dict
        terug die aanduidt wat de gelijkaardigheid
        is volgens de randen met hey gegeven stuk,
        waarbij het tweede stuk geroteerd wordt
        indien dit past. Vorm:
        {
            0 -> <scores van match met beide stukken 0x gedraaid>,
            ** Niet altijd mogelijk!
            1 -> <scores van match met het tweede stuk 90° CW gedraaid>,
            **
            2 -> <scores van match met het tweede stuk 180° CW gedraaid>,
            ** Niet altijd mogelijk!
            3 -> <scores van match met ... >
            **
        }
        """
        result = {}
        if self.img.shape[0] == self.img.shape[1]:
            # Vierkante stukken, 4 oriëntaties van
            # zowel dit stuk als het andere stuk
            # proberen en teruggeven
            for i in range(4):
                result[i] = self.match(other, accuracy, max_offset)
                other = other.rotate()
        else:
            # Rechthoekige stukken, 2 oriëntaties
            # proberen in de plaats
            for i in range(2):
                result[2 * i] = self.match(other, accuracy, max_offset)
                other = other.rotate_n(2)
        return result
    
    def match(self, other: 'TiledPiece', accuracy: int, max_offset: tuple[int, int]) -> dict:
        """
        Geeft scores in de vorm van een dict
        terug die aanduidt wat de gelijkaardigheid
        is volgens de randen met het meegegeven stuk
        in de vorm
        {
            'left': kans dat het meegegeven stuk links hoort,
            'right': kans dat het ... rechts hoort,
            'top': kans ... er boven hoort,
            'bottom': ... er onder hoort
        }
        """
        result = {}
        if self.img.shape[0] == other.img.shape[0]:
            # links matchen met de rechterkant van other
            result["left"] = self.calc_score(self.img[:, 0], other.img[:, -1], abs_threshold = accuracy, max_offset = max_offset)
            # rechts matchen met de linkerkant van other
            result["right"] = self.calc_score(self.img[:, -1], other.img[:, 0], abs_threshold = accuracy, max_offset = max_offset)
        if self.img.shape[1] == other.img.shape[1]:
            # boven matchen met de onderkant van other
            result["top"] = self.calc_score(self.img[0, :], other.img[-1, :], abs_threshold = accuracy, max_offset = max_offset)
            # onder matchen met de bovenkant van other
            result["bottom"] = self.calc_score(self.img[-1, :], other.img[0, :], abs_threshold = accuracy, max_offset = max_offset)
        return result

    def rotate(self) -> "TiledPiece":
        return TiledPiece(cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE), (self.rot + 1) % 4)
    
    def rotate_n(self, n: int) -> "TiledPiece":
        assert(n % 4 != 0)
        return TiledPiece(cv2.rotate(self.img, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][n % 4 - 1]), self.rot + (n % 4 - 1))

    def rotate_to(self, absolute_rot: int) -> "TiledPiece":
        n = (absolute_rot - self.rot) % 4 - 1
        if n == -1:
            return TiledPiece(self.img, absolute_rot)
        code = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        return TiledPiece(cv2.rotate(self.img, code[n]), absolute_rot)


    @staticmethod
    def calc_score(arr1: np.ndarray, arr2: np.ndarray, abs_threshold: int = 10, max_offset: tuple[int, int] = (-2, 2)) -> float:
        """
        Berekent de gelijkaardigheid tussen de twee gegeven arrays
        met dimensies (L), met als extra parameters de threshold
        (de maximale afwijking tussen 2 waardes) en de offset 
        (maximale afwijking in positie, zowel links (1e offset) als
        rechts (2e offset))
        """
        arr2s = np.array([np.roll(arr2, i) for i in range(max_offset[0], max_offset[1] + 1)])
        lower = np.clip(arr1.astype(np.int16) - abs_threshold, 0, 255)
        upper = np.clip(arr1.astype(np.int16) + abs_threshold, 0, 255)
        result = ((lower <= arr2s) & (arr2s <= upper)).any(axis = 0).all(axis = 1)
        return np.count_nonzero(result) / len(result)
