import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

def create_output_folder(output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def load_audio(filepath):
    # Load audio file
    y, sr = librosa.load(filepath)
    return y, sr

def generate_spectrogram(y, n_fft=2048, hop_length=512, n_mfcc=20):
    # Generate spectrogram
    spectrogram = librosa.feature.mfcc(y=y, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

def mfcc_features(spectrogram_db, filename, output_folder):
    # Calculate mean, min, and max for each MFCC
    mfcc_mean = spectrogram_db.mean(axis=1)
    mfcc_min = spectrogram_db.min(axis=1)
    mfcc_max = spectrogram_db.max(axis=1)
    mfcc_feature = np.concatenate((mfcc_mean, mfcc_min, mfcc_max))

    # Save the features to a file
    feature_filename = os.path.join(output_folder, f"{filename[:-4]}_mfcc.npy")
    np.save(feature_filename, mfcc_feature)

    return mfcc_feature

def tensors(mfcc_features_folder, output_folder):
    # Ensure the output folder exists
    create_output_folder(output_folder)

    # Load all MFCC features, normalize, and convert to tensor, then save as .npy
    for filename in os.listdir(mfcc_features_folder):
        if filename.endswith('_mfcc.npy'):
            mfcc_path = os.path.join(mfcc_features_folder, filename)
            mfcc_features = np.load(mfcc_path)

            # Normalize the MFCC features
            mfcc_normalized = (mfcc_features - np.mean(mfcc_features)) / np.std(mfcc_features)

            # Convert to tensor
            mfcc_tensor = tf.convert_to_tensor(mfcc_normalized, dtype=tf.float32)

            # Save the tensor to a file
            tensor_filename = os.path.join(output_folder, f"{filename[:-4]}_tensor.npy")
            np.save(tensor_filename, mfcc_tensor.numpy())  # Convert tensor to numpy array before saving


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
            spectrogram_db = generate_spectrogram(y, sr)
            
            # Plot and save spectrogram
            mfcc_features(spectrogram_db, filename, output_folder)

def main():
    input_folder = 'phoneme_wav'
    mfcc_features_folder = 'mfcc_features'
    tensor_output_folder = 'mfcc_tensors'
    generate_spectrograms(input_folder, mfcc_features_folder)
    tensors(mfcc_features_folder, tensor_output_folder)
    print(f"Converted MFCC features to tensors and saved to {tensor_output_folder}.")


if __name__ == "__main__":
    main()