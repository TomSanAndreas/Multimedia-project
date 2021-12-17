Project Multimedia: Legpuzzels

Auteur: Han Van Bladel, Tom Windels
Datum: 20/12/2021

-----------

OS: Windows 10 | Manjaro Linux 5.15.8-1

Versie controle:
	Python 3.9.9        | 3.10.1
	Numpy 1.21.4
	simpleaudio 1.0.4
	opencv-python 4.5.4.60
    matplotlib 3.5.1

IDE: Pycharm 2020.2 | Microsoft Visual Studio Code (VSCodium)

-----------

Bestanden/Mappen:
	Documentatie/Puzzels/Tiles_{rotated, scrambled, shuffled}/tiles_{rotated, scrambled, shuffled}_{2, 3, 4, 5}x{2, 3, 4, 5}_0{0 - 8}.png
    Audio/base.py
    Solver/base.py
    Solver/board.py
    Solver/piece.py
    Solver/scrambled_splitter.py
    Solver/types.py
    Solver/utils.py
    main.py

main.py
    python script dat de solver oproept en de gebruiker een puzzel doet kiezen om op te lossen

start commando: python main.py