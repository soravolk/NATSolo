SAMPLE_RATE = 16000 # 16000
HOP_LENGTH =  160 # SAMPLE_RATE * 32 // 1000
ONSET_LENGTH = HOP_LENGTH # SAMPLE_RATE * 32 // 1000 # 32ms
OFFSET_LENGTH = HOP_LENGTH # SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 51
MAX_MIDI = 88

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