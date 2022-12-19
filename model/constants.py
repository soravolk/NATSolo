SAMPLE_RATE = 16000 # 16000
SOLOLA_SAMPLE_RATE = 44100
BP_SAMPLE_RATE = 22050
HOP_LENGTH =  160 # SAMPLE_RATE * 32 // 1000
SOLOLA_HOP_LENGTH = 256
BP_HOP_LENGTH = (22050 * 2 - 256) / (86 * 2)
ONSET_LENGTH = HOP_LENGTH # SAMPLE_RATE * 32 // 1000 # 32ms
OFFSET_LENGTH = HOP_LENGTH # SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 40 #51
MAX_MIDI = 88
LOGIC_MIDI = 51

N_BINS = 229 # Default using Mel spectrograms
MEL_FMIN = 82 # 30
MEL_FMAX = 5000 # SAMPLE_RATE // 2

WINDOW_LENGTH = 1024 # 2048
N_FFT = 1024 # 2048

#DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOTAL_LABEL_FILES = 16
TOTAL_UNLABEL_FILES = 33
TOTAL_VALID_FILES = 2
TOTAL_TEST_FILES = 1

technique_dict = {
    0: 'no tech',
    1: 'slide',
    2: 'bend',
    3: 'trill',
    4: 'mute',
    5: 'pull',
    6: 'harmonic',
    7: 'hammer',
    8: 'tap',
    9: 'normal'
}