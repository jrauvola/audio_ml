import os
import re
import nltk
import numpy as np
import librosa
import librosa.display
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

# Function to extract phoneme MFCCs
def extract_phoneme_mfccs(y, sr, intervals, phonemes, word_idx, window_size, stride, n_mfcc=13):
    word_start, word_end = intervals[word_idx]
    phoneme_length = (word_end - word_start) // len(phonemes)
    phoneme_mfccs = []
    for i, phoneme in enumerate(phonemes):
        start_phoneme = word_start * stride + i * phoneme_length * stride
        end_phoneme = start_phoneme + phoneme_length * stride
        mfccs = librosa.feature.mfcc(y=y[start_phoneme:end_phoneme], sr=sr, n_mfcc=n_mfcc)
        phoneme_mfccs.append((phoneme, mfccs))
    return phoneme_mfccs

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

    # Extract and save MFCCs for each phoneme in each word
    all_phoneme_mfccs = []
    for word_idx, phonemes in enumerate(sentence_phonemes):
        phoneme_mfccs = extract_phoneme_mfccs(y, sr, intervals, phonemes, word_idx, window_size, stride)
        for i, (phoneme, mfccs) in enumerate(phoneme_mfccs):
            tensor = tf.convert_to_tensor(mfccs, dtype=tf.float32)
            all_phoneme_mfccs.append((phoneme, tensor))

            # Save tensor to file
            file_path = os.path.join(output_dir, f'{phoneme}.npy')
            np.save(file_path, tensor.numpy())
            print(f'Saved phoneme {phoneme} tensor to {file_path}')

    # Display results
    for phoneme, tensor in all_phoneme_mfccs:
        print(f"Phoneme: {phoneme}")
        print(f"MFCCs: {tensor.numpy()}\n")

if __name__ == "__main__":
    main()
