class PuzzlePiece:
    def __init__(self, edges, neighbours: list):
        self.neighbours = neighbours
        self.edges = edges
        # neighbours wordt met een "score" bijgehouden later,
        # adhv een edgebeeld, om zo later dan de scores
        # te optimaliseren (wat hopelijk de oplossing oplevert)
        # TODO