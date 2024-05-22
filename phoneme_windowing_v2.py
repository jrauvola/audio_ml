import os
import re
import nltk
import numpy as np
import librosa
import tensorflow as tf
from nltk.corpus import cmudict

# Ensure nltk corpus is downloaded
nltk.download('cmudict')

# Function to load and preprocess audio
def load_audio(file_path, sr=23000):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# Function to segment audio
def segment_audio(y, sr, window_size, stride):
    num_segments = (len(y) - window_size) // stride + 1
    segments = np.empty((num_segments, window_size))
    for i in range(num_segments):
        start = i * stride
        end = start + window_size
        segments[i] = y[start:end]
    return segments

# Function to detect word intervals based on energy
def detect_intervals(segments, energy_threshold):
    energy = np.sum(segments**2, axis=1)
    word_indices = np.where(energy > energy_threshold)[0]
    intervals = []
    start_idx = word_indices[0]
    for i in range(1, len(word_indices)):
        if word_indices[i] != word_indices[i - 1] + 1:
            intervals.append((start_idx, word_indices[i - 1]))
            start_idx = word_indices[i]
    intervals.append((start_idx, word_indices[-1]))
    return intervals

# Function to get phonemes for each word
def get_phonemes(word):
    pron_dict = cmudict.dict()
    word = word.lower()
    phonemes = pron_dict.get(word)
    if phonemes:
        phonemes = [[re.sub(r'\d', '', phoneme) for phoneme in variant] for variant in phonemes]
        return phonemes[0]
    else:
        return []

# Function to extract MFCCs
def extract_mfccs(y, sr, start_phoneme, end_phoneme, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y[start_phoneme:end_phoneme], sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Function to extract Mel-Spectrogram
def extract_mel_spectrogram(y, sr, start_phoneme, end_phoneme, n_mels=128, fmax=8000):
    mel_spec = librosa.feature.melspectrogram(y=y[start_phoneme:end_phoneme], sr=sr, n_mels=n_mels, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Function to extract Fundamental Frequency (F0)
def extract_f0(y, sr, start_phoneme, end_phoneme):
    pitches, magnitudes = librosa.core.piptrack(y=y[start_phoneme:end_phoneme], sr=sr, n_fft=2048, hop_length=512)
    f0 = [np.max(pitches[:, t]) if np.max(magnitudes[:, t]) > 0.1 else 0 for t in range(pitches.shape[1])]
    return f0

# Function to extract Harmonic and Noise Components
def extract_harmonic_noise(y, start_phoneme, end_phoneme):
    harmonic, percussive = librosa.effects.hpss(y[start_phoneme:end_phoneme])
    noise = y[start_phoneme:end_phoneme] - harmonic
    return harmonic, noise

# Function to extract Formant Frequencies
def extract_formants(y, sr, start_phoneme, end_phoneme):
    lpc_coeffs = librosa.lpc(y[start_phoneme:end_phoneme], order=16)
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]
    angles = np.angle(roots)
    frequencies = angles * (sr / (2 * np.pi))
    formants = np.sort(frequencies)
    return formants

# Function to extract Amplitude Envelope
def extract_amplitude_envelope(y, sr, start_phoneme, end_phoneme):
    amplitude_envelope = librosa.onset.onset_strength(y=y[start_phoneme:end_phoneme], sr=sr)
    return amplitude_envelope

# Function to extract features for each phoneme
def extract_phoneme_features(y, sr, intervals, phonemes, word_idx, window_size, stride, output_dir):
    word_start, word_end = intervals[word_idx]
    phoneme_length = (word_end - word_start) // len(phonemes)
    phoneme_features = []
    for i, phoneme in enumerate(phonemes):
        start_phoneme = word_start * stride + i * phoneme_length * stride
        end_phoneme = start_phoneme + phoneme_length * stride

        # Extract all features
        mfccs = extract_mfccs(y, sr, start_phoneme, end_phoneme)
        mel_spec_db = extract_mel_spectrogram(y, sr, start_phoneme, end_phoneme)
        f0 = extract_f0(y, sr, start_phoneme, end_phoneme)
        harmonic, noise = extract_harmonic_noise(y, start_phoneme, end_phoneme)
        formants = extract_formants(y, sr, start_phoneme, end_phoneme)
        amplitude_envelope = extract_amplitude_envelope(y, sr, start_phoneme, end_phoneme)

        # Save features
        save_feature(output_dir, phoneme, 'mfccs', mfccs)
        save_feature(output_dir, phoneme, 'mel_spec', mel_spec_db)
        save_feature(output_dir, phoneme, 'f0', f0)
        save_feature(output_dir, phoneme, 'harmonic', harmonic)
        save_feature(output_dir, phoneme, 'noise', noise)
        save_feature(output_dir, phoneme, 'formants', formants)
        save_feature(output_dir, phoneme, 'amplitude_envelope', amplitude_envelope)

        phoneme_features.append({
            'phoneme': phoneme,
            'mfccs': mfccs,
            'mel_spec_db': mel_spec_db,
            'f0': f0,
            'harmonic': harmonic,
            'noise': noise,
            'formants': formants,
            'amplitude_envelope': amplitude_envelope
        })

    return phoneme_features

# Function to save features in the respective folder
def save_feature(output_dir, phoneme, feature_name, feature_data):
    feature_dir = os.path.join(output_dir, feature_name)
    os.makedirs(feature_dir, exist_ok=True)
    file_path = os.path.join(feature_dir, f'{phoneme}.npy')
    np.save(file_path, feature_data)
    print(f'Saved {feature_name} for phoneme {phoneme} to {file_path}')

# Main function
def main():
    file_path = '/Users/paulchang/Desktop/audio_ml/hello_how_are_u.wav'
    output_dir = '/Users/paulchang/Desktop/audio_ml/'

    # Load and preprocess audio
    y, sr = load_audio(file_path)
    window_size = int(0.025 * sr)  # 25 ms
    stride = int(0.010 * sr)       # 10 ms

    # Segment audio
    segments = segment_audio(y, sr, window_size, stride)

    # Detect intervals
    energy_threshold = np.max(np.sum(segments**2, axis=1)) * 0.1
    intervals = detect_intervals(segments, energy_threshold)

    # Prepare sentence and get phonemes
    sentence = "Hello how are you."
    words = [re.sub(r'[^a-zA-Z]', '', word) for word in sentence.split()]
    sentence_phonemes = [get_phonemes(word) for word in words]

    # Extract and save features for each phoneme in each word
    for word_idx, phonemes in enumerate(sentence_phonemes):
        extract_phoneme_features(y, sr, intervals, phonemes, word_idx, window_size, stride, output_dir)

if __name__ == "__main__":
    main()
