import threading
import random as r
import numpy as np
import simpleaudio as sa

class Audio(threading.Thread):
    def __init__(self, notes):
        super(Audio, self).__init__()
        self.notes = notes
        self.is_playing = False
        self.play_obj = None

    def play(self, time: float = .25, fs: int = 8000) -> None:
        self.time = time
        self.fs = fs
        self.is_playing = True
        self.start()
    
    def stop(self) -> None:
        if self.play_obj is not None:
            self.play_obj.stop()

    def run(self):
        freqs = []
        aantal_noten = len(self.notes)
        base_maj_freq = Audio.frequenties(self.notes[0])
        for i in range(aantal_noten):
            f = Audio.frequenties(self.notes[i])
            s = f / base_maj_freq
            _, yt = Audio.karplus_strong_met_uitrekken(f, self.fs, self.time, s)
            freqs.extend(yt)
        # resultaat afspelen
        # Ensure that highest value is in 16-bit range
        audio = freqs/np.max(np.abs(freqs))
        audio = audio * (2 ** 15 - 1)
        # Convert to 16-bit data
        audio = audio.astype(np.int16)
        # Start playback
        self.play_obj = sa.play_buffer(audio, 1, 2, self.fs)
        # Wait for playback to finish before exiting
        self.play_obj.wait_done()
        self.is_playing = False
    
    @staticmethod
    def frequenties(note: str = "C1", print_debug: bool = False) -> float:
        base_freq = 32.703
        noten = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        # Omzetten van omschrijving naar parameters voor functie
        octave = int(note[-1]) - 1
        try:
            index = noten.index(note[:-1])
        except ValueError:
            print(f"Ongeldige noot \"{note}\" opgegeven!")
            return base_freq
        # Berekenen resulterende frequentie
        # Voor debug-informatie
        if print_debug:
            print(
                f"Note \"{note}\" -> octave {octave + 1} ; index {index} ; freq {base_freq * 2 ** (index / 12 + octave)}")
        return base_freq * 2 ** (index / 12 + octave)

    @staticmethod
    def karplus_strong_met_uitrekken(f: int, fs: int, time: float, s: int) -> tuple:
        # bepalen van aantal elementen p afh van f en fs
        p = int(fs / f - .5)
        # creeeren van initiele -1,1 buffer met p elementen
        yt = [1 - 2 * r.randint(0, 1) for i in range(p + 1)]
        # yt vervolledigen met aantal samples uit de toon zelf
        prob = 1 - 1 / s
        for i in range(int(fs * time)):
            yt.append(yt[i] if r.random() < prob else (yt[i] + yt[i + 1]) / 2)
        return yt[:p], yt[p + 1:]

#----Gebruik-----
#--notes definieren--
fail_notes = ['C3', 'F3', 'F3', 'F3', 'E3', 'D3', 'C3', 'G2', 'G2', 'C2']
victory_notes = ['G2', 'C3', 'E3', 'G3', 'C4', 'E4', 'G4', 'E4','G#2', 'C3', 'D#3', 'G#3', 'C4', 'D#4', 'G#4', 'D#4',
                 'A#2', 'D3', 'F3', 'A#3', 'D4', 'F4', 'A#4', 'A#4','A#4','C5']
wait_notes = ['E4','E4','C4', 'A3', 'A3', 'D4', 'D4', 'D4', 'F#4', 'F#4', 'G4', 'A4', 'G4', 'G4','G4', 'D4', 'C4', 'E4',
              'E4', 'E4','D4', 'D4', 'E4', 'D4','E4','E4','C4', 'A3', 'A3', 'D4', 'D4', 'D4', 'F#4', 'F#4', 'G4', 'A4',
              'G4', 'G4','G4', 'D4', 'C4', 'E4','E4', 'E4','D4', 'D4', 'E4', 'D4']
