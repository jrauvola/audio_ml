import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def create_output_folder(output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def load_audio(filepath):
    # Load audio file
    y, sr = librosa.load(filepath)
    return y, sr

def generate_spectrogram(y, sr):
    # Generate spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

def plot_and_save_spectrogram(spectrogram, sr, filename, output_folder):
    # Plot and save spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    output_filepath = os.path.join(output_folder, os.path.splitext(filename)[0] + '_spectrogram.png')
    plt.savefig(output_filepath)
    plt.close()

def generate_spectrograms(input_folder, output_folder):
    # Create output folder if it doesn't exist
    create_output_folder(output_folder)
    
    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            # Load audio file
            filepath = os.path.join(input_folder, filename)
            y, sr = load_audio(filepath)
            
            # Generate spectrogram
            spectrogram = generate_spectrogram(y, sr)
            
            # Plot and save spectrogram
            plot_and_save_spectrogram(spectrogram, sr, filename, output_folder)

def main():
    input_folder = 'phonemes_wav'
    output_folder = 'spectrograms'
    generate_spectrograms(input_folder, output_folder)

if __name__ == "__main__":
    main()
