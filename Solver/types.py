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

    @staticmethod
    def is_tiled(t: int) -> bool:
        return t == TILES_ROTATED or t == TILES_SCRAMBLED or t == TILES_SHUFFLED
    
    @staticmethod
    def is_jigsaw(t: int) -> bool:
        return t == JIGSAW_ROTATED or t == JIGSAW_SCRAMBLED or t == JIGSAW_SHUFFLED
    
    