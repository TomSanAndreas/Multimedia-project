import threading
import random as r
import numpy as np
import simpleaudio as sa

class Audio(threading.Thread):
    def __init__(self):
        super(Audio, self).__init__()

    def speel_af(sample, rate):
        # Ensure that highest value is in 16-bit range
        audio = sample / np.max(np.abs(sample))
        audio = audio * (2 ** 15 - 1)
        # Convert to 16-bit data
        audio = audio.astype(np.int16)
        # Start playback
        play_obj = sa.play_buffer(audio, 1, 2, rate)
        # Wait for playback to finish before exiting
        play_obj.wait_done()

    def frequenties(note: str = "C1", print_debug: bool = False) -> float:
        base_freq = 32.703
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        # Omzetten van omschrijving naar parameters voor functie
        octave = int(note[-1]) - 1
        try:
            index = notes.index(note[:-1])
        except ValueError:
            print(f"Ongeldige noot \"{note}\" opgegeven!")
            return base_freq
        # Berekenen resulterende frequentie
        # Voor debug-informatie
        if print_debug:
            print(
                f"Note \"{note}\" -> octave {octave + 1} ; index {index} ; freq {base_freq * 2 ** (index / 12 + octave)}")
        return base_freq * 2 ** (index / 12 + octave)

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

    @staticmethod
    def puzzel_sound_play(self, notes, aantal_noten, time: float = .25, fs: int = 8000) -> None:
        c_maj = []
        base_maj_freq = self.frequenties(notes[0])
        for i in range(aantal_noten):
            f = self.frequenties(notes[i])
            s = f / base_maj_freq
            _, yt = self.karplus_strong_met_uitrekken(f, fs, time, s)
            c_maj.extend(yt)
        # resultaat opslaan indien nodig
        # Write(c_maj, fs, "Fail")
        # resultaat afspelen
        self.speel_af(c_maj, fs)

#Testen
fail_sound = ['C3', 'F3', 'F3', 'F3', 'E3', 'D3', 'C3', 'G2', 'G2', 'C2']
aud = Audio()
Audio.puzzel_sound_play(aud, fail_sound, 10)
