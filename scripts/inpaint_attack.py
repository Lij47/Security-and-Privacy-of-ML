import torch
import random
import librosa
from PIL import Image
import numpy as np
from datasets import load_dataset
from IPython.display import Audio
from tqdm import tqdm
import os
from audiodiffusion import AudioDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)

model_id = "teticio/audio-diffusion-breaks-256"

audio_diffusion = AudioDiffusion(model_id=model_id)
mel = audio_diffusion.pipe.mel

dataset_train = torch.load('dataset_train.pt')
dataset_non_train = torch.load('dataset_non_train.pt')

def random_window(x):
    return np.random.randn()

def variance(x):
    return np.var(x)

def entropy(x):
    _ , counts = np.unique(x, return_counts=True)
    probabilities = counts / x.shape[-1]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def dynamic_range(x):
    return(10 * np.log10(np.max(x) / np.min(x)))

def top_k_windows(arr, window_size, metric, k, sort_max=True):
    values = []
    for i in range(window_size, arr.shape[-1]-window_size, window_size*3):
        window = arr[i:i+window_size]
        values.append((metric(window), i))
    sorted_list = sorted(values, key=lambda x: x[0], reverse=sort_max)
    return sorted_list[:k]

def measure_norm(input_data, inpaints, mask):
    diff = ((input_data - inpaints) * (mask == False))
    norm = np.sum(diff**2, axis=(1,2)) / np.sum(mask[0] == False)
    norm = np.sqrt(norm) 
    norm = np.mean(norm) 
    return norm

audio_length = 130560
audio_length_sec = 5
gap_size = 0.02
window_size = int(audio_length/audio_length_sec * gap_size)
num_gaps = 10
print((window_size*num_gaps)/audio_length)
num_audio = 500
num_inpaints = 5

dataset_train_permutations = np.random.permutation(len(dataset_train["train"]))
dataset_non_train_permutations = np.random.permutation(len(dataset_non_train["train"]))
dataset_permutations_dict = {
    "dataset_train_permutations": dataset_train_permutations,
    "dataset_non_train_permutations": dataset_non_train_permutations,
}
torch.save(dataset_permutations_dict, "dataset_permutations.pt")


dataset_permutations_dict = torch.load('dataset_permutations.pt')
dataset_train_permutations = dataset_permutations_dict['dataset_train_permutations']
dataset_non_train_permutations = dataset_permutations_dict['dataset_non_train_permutations']


print("Entropy")
train_entropy_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, entropy, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    train_entropy_norm_list.append(norm)

torch.save(train_entropy_norm_list, "norm_data/train_entropy_norm_list.pt")

non_train_entropy_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_non_train_permutations[i].item()
    image = dataset_non_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, entropy, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    non_train_entropy_norm_list.append(norm)

torch.save(non_train_entropy_norm_list, "norm_data/non_train_entropy_norm_list.pt")

print("Dynamic Range")
train_dynamic_range_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, dynamic_range, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    train_dynamic_range_norm_list.append(norm)

torch.save(train_dynamic_range_norm_list, "norm_data/train_dynamic_range_norm_list.pt")

non_train_dynamic_range_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_non_train_permutations[i].item()
    image = dataset_non_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, dynamic_range, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    non_train_dynamic_range_norm_list.append(norm)

torch.save(non_train_dynamic_range_norm_list, "norm_data/non_train_dynamic_range_norm_list.pt")

print("Variance")
train_variance_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, variance, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    train_variance_norm_list.append(norm)

torch.save(train_variance_norm_list, "norm_data/train_variance_norm_list.pt")
non_train_variance_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_non_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, variance, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    non_train_variance_norm_list.append(norm)

torch.save(non_train_variance_norm_list, "norm_data/train_variance_norm_list.pt")

print("Random")
train_random_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, random_window, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    train_random_norm_list.append(norm)

torch.save(train_random_norm_list, "norm_data/train_random_norm_list.pt")

non_train_random_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_non_train_permutations[i].item()
    image = dataset_non_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, random_window, num_gaps)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    non_train_random_norm_list.append(norm)

torch.save(non_train_random_norm_list, "norm_data/non_train_random_norm_list.pt")

print("min entropy")
train_entropy_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, entropy, num_gaps, False)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    train_entropy_norm_list.append(norm)

torch.save(train_entropy_norm_list, "norm_data_min/train_entropy_norm_list.pt")

non_train_entropy_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_non_train_permutations[i].item()
    image = dataset_non_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, entropy, num_gaps, False)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    non_train_entropy_norm_list.append(norm)

torch.save(non_train_entropy_norm_list, "norm_data_min/non_train_entropy_norm_list.pt")

print("min dynamic range")
train_dynamic_range_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, dynamic_range, num_gaps, False)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    train_dynamic_range_norm_list.append(norm)

torch.save(train_dynamic_range_norm_list, "norm_data_min/train_dynamic_range_norm_list.pt")

non_train_dynamic_range_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_non_train_permutations[i].item()
    image = dataset_non_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, dynamic_range, num_gaps, False)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    non_train_dynamic_range_norm_list.append(norm)

print("Variance min")
train_variance_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_train_permutations[i].item()
    image = dataset_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, variance, num_gaps, False)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    train_variance_norm_list.append(norm)

torch.save(train_variance_norm_list, "norm_data_min/train_variance_norm_list.pt")

non_train_variance_norm_list = []
for i in tqdm(range(num_audio)):
    i = dataset_non_train_permutations[i].item()
    image = dataset_non_train['train'][i]['image']
    audio = mel.image_to_audio(image)
    mask = np.ones((audio.shape[0]), dtype=bool)
    top_k = top_k_windows(audio, window_size, variance, num_gaps, False)
    for tk in top_k:
        mask[tk[1]:tk[1]+window_size] = False
    image_mask = mask[::510]
    image_mask = image_mask.reshape(1, 1, 256)
    
    masked_audio = audio * mask
    
    inpaint_image = audio_diffusion.generate_spectrogram_and_audio_from_audio(
        raw_audio=masked_audio,
        batch_size=num_inpaints,
        start_step=900,
        step_generator=torch.Generator(device="cuda"))

    image = np.array(image)
    image = image.reshape(1, 256, 256)
    norm = measure_norm(image, inpaint_image, image_mask)
    non_train_variance_norm_list.append(norm)

torch.save(non_train_variance_norm_list, "norm_data_min/non_train_variance_norm_list.pt")