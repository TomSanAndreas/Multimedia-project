import numpy as np
import simpleaudio as sa
from scipy.io import wavfile
from matplotlib import pyplot as plt
import random as r
import re
from scipy.io.wavfile import write

# Algemene (en gegeven) functies

def Audio(sample, rate):
    # Ensure that highest value is in 16-bit range
    audio = sample / np.max(np.abs(sample))
    audio = audio * (2 ** 15 - 1)
    # Convert to 16-bit data
    audio = audio.astype(np.int16)
    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, rate)
    # Wait for playback to finish before exiting
    play_obj.wait_done()

def Write(audio, rate, name):
    # Audio moet niet meer worden opgeslagen
    # Indien wel, mogen deze lijnen uit commentaar worden gehaald
    audio_temp = np.array(audio) if isinstance(audio, list) else audio
    wavfile.write(f"output/{name}.wav", rate, audio_temp)
    #pass

def karplus_strong(f: int, fs: int, time: float) -> tuple:
    # bepalen van aantal elementen p afh van f en fs
    p = int(fs / f - .5)
    # creeeren van initiele -1,1 buffer met p elementen
    yt = [1 - 2 * r.randint(0, 1) for i in range(p + 1)]
    # yt vervolledigen met aantal samples uit de toon zelf
    for i in range(int(fs * time)):
        yt.append((yt[i + 1] + yt[i]) / 2)
    return yt[:p], yt[p + 1:]

# parameters gebruikt voor opgave 2, die als constantes gebruikt worden
base_freq = 32.703
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def frequenties(note: str = "C1", print_debug: bool = False) -> float:
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
        print(f"Note \"{note}\" -> octave {octave + 1} ; index {index} ; freq {base_freq * 2 ** (index / 12 + octave)}")
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

def puzzel_sound_play(notes,aantal_noten,  time: float = .25, fs: int = 8000) -> None:
    c_maj = []
    base_maj_freq = frequenties(notes[0])
    for i in range(aantal_noten):
        f = frequenties(notes[i])
        s = f / base_maj_freq
        _, yt = karplus_strong_met_uitrekken(f, fs, time, s)
        c_maj.extend(yt)
    # resultaat opslaan indien nodig
    #Write(c_maj, fs, "Fail")
    # resultaat afspelen
    Audio(c_maj, fs)

def play_note(note: str = "C1", fs: int = 8000, time: float = 1.0, s: int = 1) -> list:
    f = frequenties(note)
    _, yt = karplus_strong_met_uitrekken(f, fs, time, s)
    return yt

"""
import re
note = "16f#6"
duration_pattern = re.compile("^[0-9]{1,2}")

pitch_pattern = re.compile("[a-zA-Z]#*")
octave_pattern = re.compile("[0-9]$")
special_pattern = re.compile("[.]")
duration = duration_pattern.findall(note)
pitch = pitch_pattern.findall(note)[0].lower()
octave = octave_pattern.findall(note)
special_duration = special_pattern.findall(note)
"""

"""RTTTL"""
duration_pattern = re.compile("^[0-9]{1,2}")
pitch_pattern = re.compile("[a-zA-Z]#*")
octave_pattern = re.compile("[0-9]$")
special_pattern = re.compile("[.]")

def rtttl_functie(source: str, name: str, fs: int = 8000) -> None:
    txt_data = source.lower().split("rtttl = ")
    file_data = txt_data[0].split(" = ")
    note_data = [note.strip() for note in txt_data[1].replace("\n", "").split(",")]
    default_octave = int(file_data[1][:file_data[1].index(',')])
    default_duration = int(file_data[2][:file_data[2].index(',')])
    bpm = int(file_data[3])
    result = []
    fc2 = frequenties("C2")
    fade = [(i / (bpm / 60 / 128 * fs)) ** .5 for i in range(int(bpm / 60 / 128 * fs))]
    for note in note_data:
        # data van note inlezen
        duration = duration_pattern.findall(note)
        pitch = pitch_pattern.findall(note)
        octave = octave_pattern.findall(note)
        special_duration = special_pattern.findall(note)
        # data van note verwerken
        octave = default_octave if len(octave) == 0 else int(octave[0])
        duration = 1 / default_duration if len(duration) == 0 else 1 / int(duration[0]) if len(special_duration) == 0 else 1.5 / int(duration[0])
        pitch = pitch[0].upper()
        # verwerkte data omzetten naar bruikbare parameters voor karplus strong
        f = 0 if pitch[0] == "P" else frequenties(f"{pitch}{octave}")
        time = duration * bpm / 60
        if f != 0:
            s = f / fc2
            # noot synthetiseren
            data = karplus_strong_met_uitrekken(f, fs, time, s)[1]
            # fadein effect toevoegen
            # data = [data[i] if i >= len(fade) else data[i] * fade[i] for i in range(len(data))]
            # toevoegen aan resultaat
            result.extend(data)
        else:
            result.extend([0 for i in range(int(fs * time))])
    # resultaat opslaan indien nodig
    Write(result, fs, f"opdracht_6_{name}")
    # resultaat afspelen
    Audio(result, fs)

"""Piano test - functies"""
def get_wave(freq, duration=0.5):
    '''
    Function takes the "frequecy" and "time_duration" for a wave
    as the input and returns a "numpy array" of values at all points
    in time
    '''
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    return wave

def get_piano_notes():
    '''
    Returns a dict object for all the piano
    note's frequencies
    '''
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 261.63  # Frequency of Note C4
    note_freqs = {octave[i]: base_freq * pow(2, (i / 12)) for i in range(len(octave))}
    note_freqs[''] = 0.0  # silent note
    return note_freqs

def get_song_data(music_notes):
    '''
    Function to concatenate all the waves (notes)
    '''
    note_freqs = get_piano_notes() # Function that we made earlier
    song = [get_wave(note_freqs[note]) for note in music_notes.split('-')]
    song = np.concatenate(song)
    return song

if __name__ == "__main__":
    """Test Muziek"""
    fail_sound = ['C3', 'F3', 'F3', 'F3', 'E3', 'D3', 'C3', 'G2', 'G2', 'C2']
    puzzel_sound_play(fail_sound, 10)
    start_sound = ['E4', 'G4', 'E5', 'C5', 'E5', 'G5']
    puzzel_sound_play(start_sound, 6, time=0.13)
    vicotry_sound1 = ['G2', 'C3', 'E3', 'G3', 'C4', 'E4', 'G4', 'E4']
    vicotry_sound2 = ['G#2', 'C3', 'D#3', 'G#3', 'C4', 'D#4', 'G#4', 'D#4']
    vicotry_sound3 = ['A#2', 'D3', 'F3', 'A#3', 'D4', 'F4', 'A#4', 'A#4','A#4','C5']
    puzzel_sound_play(vicotry_sound1, 8, time=0.2)
    puzzel_sound_play(vicotry_sound2, 8, time=0.2)
    puzzel_sound_play(vicotry_sound3, 10, time=0.2)

    """Piano-functies testen"""
    samplerate = 44100
    # a_wave = get_wave(440, 1)
    # music_notes = 'C-C-G-G-A-A-G--F-F-E-E-D-D-C--G-G-F-F-E-E-D--G-G-F-F-E-E-D--C-C-G-G-A-A-G--F-F-E-E-D-D-C'
    # data = get_song_data(music_notes)
    # write('twinkle.wav', samplerate, data.astype(np.int16))