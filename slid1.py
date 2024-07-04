import os
import librosa
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


nfiltbank = 13
nComponents = 8

# Function to preprocess audio files and extract MFCC features
def preprocess_and_extract_mfcc(audio_file):
    signal, fs = librosa.load(audio_file, sr=None)
    
    # Silence removal
    signal = librosa.effects.trim(signal, top_db=20)[0]
    window_size_ms = 25
    window_size_samples = int((window_size_ms / 1000) * fs)
    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=nfiltbank, hop_length=window_size_samples)
    
    return mfcc_features.T  # Transpose for LBG input shape

# LBG Algorithm for Vector Quantization (Codebook Generation)
def lbg(X, k):
    # Initialize codebook with the mean of input data
    codebook = np.mean(X, axis=0, keepdims=True)
    
    for i in range(int(np.log2(k))):
        new_codebook = np.repeat(codebook, 2, axis=0)
        noise = 0.01 * np.random.randn(new_codebook.shape[0], new_codebook.shape[1])  # Generate noise matrix
        new_codebook += noise
        distortions = cdist(X, new_codebook, 'euclidean')
        indices = np.argmin(distortions, axis=1)
        codebook = np.array([np.mean(X[indices == j], axis=0) for j in range(2 ** (i + 1))])

    return codebook

# Define directories and language labels
data_directory = "C:/Users/Akash/Downloads/S2/DataSet1"
languages = {
             "Arabic": os.path.join(data_directory, "Arabic"),
             "English": os.path.join(data_directory, "English"),
             "Spanish": os.path.join(data_directory, "Spanish"),
             "French": os.path.join(data_directory, "French"),
             "Urdu": os.path.join(data_directory, "Urdu"),
             "Malay": os.path.join(data_directory, "Malay"),
             "German": os.path.join(data_directory, "German"),
             "Persian": os.path.join(data_directory, "Persian")
             }

# Initialize codebooks for each language using LBG algorithm
language_codebooks = {}
for language, directory in languages.items():
    mfcc_samples = np.array([])
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            audio_file = os.path.join(directory, file_name)
            mfcc_features = preprocess_and_extract_mfcc(audio_file)
            if mfcc_samples.size == 0:
                mfcc_samples = mfcc_features
            else:
                mfcc_samples = np.vstack((mfcc_samples, mfcc_features))
    X_train, X_test = train_test_split(mfcc_samples, test_size=0.3, random_state=42)
    codebook = lbg(X_train, nComponents)
    language_codebooks[language] = codebook

# Initialize GMM models for each language and codebook
language_models_gmm = {}
for language, codebook in language_codebooks.items():
    gmm = GaussianMixture(n_components=nComponents)
    gmm.fit(codebook)
    language_models_gmm[language] = gmm

# Function to identify language using GMM models with codebooks
def identify_language_gmm(audio_file):
    mfcc_features = preprocess_and_extract_mfcc(audio_file)
    likelihoods = {language: model.score(mfcc_features) for language, model in language_models_gmm.items()}
    identified_language = max(likelihoods, key=likelihoods.get)
    return identified_language

# Test the model on the test set and calculate accuracy
true_labels = []
predicted_labels = []
correct_matches_gmm = 0
total_tests_gmm = 0

for language, directory in languages.items():
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            audio_file = os.path.join(directory, file_name)
            identified_language_gmm = identify_language_gmm(audio_file)
            true_labels.append(language)
            predicted_labels.append(identified_language_gmm)
            print(f"Identified Language in {file_name} is: {identified_language_gmm}")
            if identified_language_gmm.lower() == language.lower():
                correct_matches_gmm += 1
            total_tests_gmm += 1

print(correct_matches_gmm," perfectly matched out of ",total_tests_gmm)
accuracy_gmm = (correct_matches_gmm / total_tests_gmm) * 100
print(f"Accuracy: {accuracy_gmm}%")

# Manually calculate confusion matrix and classification metrics
labels = list(languages.keys())
conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

for true_label, pred_label in zip(true_labels, predicted_labels):
    true_index = labels.index(true_label)
    pred_index = labels.index(pred_label)
    conf_matrix[true_index, pred_index] += 1

# Calculate precision, recall, F1 score, and accuracy for each language
precision = {}
recall = {}
f1_score = {}
accuracy = {}
accuracies = []

for i, label in enumerate(labels):
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = conf_matrix[i, :].sum() - tp
    tn = conf_matrix.sum() - (tp + fp + fn)
    
    precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall[label] = tp / (fn + tp) if (fn + tp) > 0 else 0
    f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0
    accuracy[label] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    accuracies.append(accuracy)

print("Confusion Matrix: ")
print(conf_matrix)
for label in labels:
    # print(f"{label} - Precision: {precision[label]:.2f}, Recall: {recall[label]:.2f}, F1 Score: {f1_score[label]:.2f}, Accuracy: {accuracy[label]:.2f}")
    print(f"{label} - Precision(%): {round(precision[label]*100)}, Recall(%): {round(recall[label]*100)}, F1 Score(%): {round(f1_score[label]*100)}, Accuracy(%): {round(accuracy[label]*100)}")

# Visualize the confusion matrix using heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Reds')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title('Confusion Matrix Heatmap')
plt.show()










# import numpy as np
# import librosa
# from scipy.linalg import sqrtm

# # Load the audio file and extract MFCC features
# audio_file = 'Speaker_0001_00000.wav'
# y, sr = librosa.load(audio_file, sr=None)
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Assuming 13 MFCC coefficients
# print("MFCCs:", mfccs)
# print(mfccs.shape)

# # Concatenate MFCCs, delta, and double delta features to form MMFCCs
# delta = librosa.feature.delta(mfccs)
# double_delta = librosa.feature.delta(mfccs, order=2)
# mmfccs = np.vstack([mfccs, delta, double_delta])

# # Reshape MMFCCs to match the dimensionality used in the code snippet (60-dimensional)
# mmfccs_reshaped = mmfccs.reshape(-1, mmfccs.shape[1])

# # Define the GMM-UBM model parameters (replace with actual values)
# n_components = 512  # Number of mixture components in the UBM
# dim = 60  # Dimensionality of MFCC features (including delta and double delta)

# # Simulated Baum-Welch statistics (replace with actual computations)
# N_p = np.random.rand(n_components)
# F_p = np.random.rand(n_components, dim)

# # Compute the within-class scatter matrix Sw
# Sw = np.zeros((dim, dim))
# for c in range(n_components):
#     Sw += N_p[c] * np.outer(F_p[c], F_p[c])

# # Compute the total variability matrix T
# T = np.linalg.inv(sqrtm(Sw))

# # Simulate a supervector M (replace with actual computations)
# M = mmfccs_reshaped.flatten()  # Assuming mmfccs_reshaped contains actual MFCCs data

# # Generate i-vectors for each supervector M
# i_vectors = []
# for r in range(M.shape[0] // dim):
#     M_r = M[r * dim: (r + 1) * dim]
#     i_vector = np.dot(T, M_r - np.mean(M_r))
#     i_vectors.append(i_vector)

# # Convert i-vectors to numpy array
# i_vectors = np.array(i_vectors)
# print("Computed i-vectors:")
# # print(i_vectors)
# print(i_vectors.shape)
# # print(len(i_vectors))

