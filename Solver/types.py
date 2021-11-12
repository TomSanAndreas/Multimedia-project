class Types:
    JIGSAW_ROTATED = 1
    JIGSAW_SCRAMBLED = 2
    JIGSAW_SHUFFLED = 3
    TILES_ROTATED = 4
    TILES_SCRAMBLED = 5
    TILES_SHUFFLED = 6
    GUESS = 99

    @staticmethod
    def convert_input_to_type(desc: str) -> int:
        return Types.GUESS
