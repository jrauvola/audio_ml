import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import os

# Define parameters for spectrogram calculation
n_fft = 2048  # Number of samples per Fast-Fourier Transform
hop_length = 512  # Number of FFT shifts (in intervals)

def m4a_to_wav(m4a_file, wav_dir):
    # Load the M4A audio file
    audio = AudioSegment.from_file(m4a_file, format="m4a")

    # Construct the WAV file name based on the M4A file name
    wav_files = os.path.join(wav_dir, os.path.basename(m4a_file).replace(".m4a", ".wav"))

    # Export the audio to WAV format
    audio.export(wav_file, format="wav")


def log_spectrogram(wav_files):
    # Create a new folder to store spectrogram plots
    output_folder = os.path.join(folder_path, 'Spectrogram_Plots')
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
    
    for wav_file in wav_files:
        # Load audio file
        file_path = os.path.join(folder_path, wav_file)
        signal, sr = librosa.load(file_path, sr=22050)  # Load with a sample rate of 22050 Hz
        
        # Calculate Short-Time Fourier Transform (STFT)
        stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
        
        # Calculate spectrogram and convert to logarithmic scale
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        
        # Plot and save the spectrogram in the new folder
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
        
        # Save the plot in the new folder
        plot_file_name = os.path.splitext(wav_file)[0] + '_spectrogram.png'
        plot_file_path = os.path.join(output_folder, plot_file_name)
        plt.savefig(plot_file_path)
        
        # Close the plot to free up resources
        plt.close()


# MFFCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13) # 13 to 40 is common for music

def main():
    current_directory = os.getcwd()

    directory = current_directory + '/Phoneme Recordings'

    # Set path to folder containing .wav files
    folder_path = directory

    # Get list of .wav files in the folder
    m4a_files = [f for f in os.listdir(folder_path) if f.endswith('.m4a')]

    for i in len(m4a_files):
        m4a_to_wav(m4a_files[i])
        

log_spectrogram(wav_files)