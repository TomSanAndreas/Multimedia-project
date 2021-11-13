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
        return t == Types.TILES_ROTATED or t == Types.TILES_SCRAMBLED or t == Types.TILES_SHUFFLED
    
    @staticmethod
    def is_jigsaw(t: int) -> bool:
        return t == Types.JIGSAW_ROTATED or t == Types.JIGSAW_SCRAMBLED or t == Types.JIGSAW_SHUFFLED
    
    