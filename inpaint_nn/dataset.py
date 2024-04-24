import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from tqdm import tqdm

from PIL import Image
import numpy as np
from datasets import load_dataset
from IPython.display import Audio
from tqdm import tqdm
import os
from audiodiffusion import AudioDiffusion

def random_window(x):
    return np.random.randn()

def entropy(x):
    _ , counts = np.unique(x, return_counts=True)
    probabilities = counts / x.shape[-1]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def dynamic_range(x):
    return(10 * np.log10(np.max(x) / np.min(x)))

def variance(x):
    return np.var(x)

def top_k_windows(arr, window_size, metric, k, sort_max=True):
    values = []
    for i in range(window_size, arr.shape[-1]-window_size, window_size*3):
        window = arr[i:i+window_size]
        values.append((metric(window), i))
    sorted_list = sorted(values, key=lambda x: x[0], reverse=sort_max)
    return sorted_list[:k]


class AudioData(Dataset):
    def __init__(self, dataset_config):
        '''
            initialize the dataset
        '''

        dataset_train = torch.load('dataset_train.pt')
        dataset_non_train = torch.load('dataset_non_train.pt')
        dataset_permutations_dict = torch.load('dataset_permutations.pt')
        dataset_train_permutations = dataset_permutations_dict['dataset_train_permutations']
        dataset_non_train_permutations = dataset_permutations_dict['dataset_non_train_permutations']
        self.audio_diffusion = AudioDiffusion(model_id="teticio/audio-diffusion-breaks-256", cuda=True)
        self.mel = self.audio_diffusion.pipe.mel
        self.audio_length = 130560
        self.audio_length_sec = 5
        self.gap_size = 0.02
        self.window_size = int(self.audio_length/self.audio_length_sec * self.gap_size)
        self.num_gaps = 10
        self.num_inpaints = 1

        if dataset_config["load_dir"] is not None:
            dataset = torch.load(dataset_config["load_dir"])
            self.features = dataset["features"]
            self.labels = dataset["labels"]
        else:
            self.features = []
            self.labels = []
            for i in tqdm(range(dataset_config["start_idx"], dataset_config["start_idx"]+dataset_config["num_train_examples"])):
                image = dataset_train["train"][dataset_train_permutations[i].item()]["image"]
                audio = self.mel.image_to_audio(image)
                image = np.array(image)

                mask = np.ones((audio.shape[0]), dtype=bool)
                top_k = top_k_windows(audio, self.window_size, entropy, self.num_gaps, sort_max=False)
                for tk in top_k:
                    mask[tk[1]:tk[1]+self.window_size] = False
                image_mask = mask[::510]
                image_mask = image_mask.reshape(1, 1, 256)
                
                masked_audio = audio * mask
                
                inpaint_image = self.audio_diffusion.generate_spectrogram_and_audio_from_audio(
                    raw_audio=masked_audio,
                    batch_size=self.num_inpaints,
                    start_step=900,
                    step_generator=torch.Generator(device="cuda"))

                image = image.reshape((1, 256, 256)) 
                image_mask = np.tile(image_mask, (inpaint_image.shape[0], inpaint_image.shape[1], 1))
                inpaint_feature = inpaint_image[image_mask==False]
                image_feature = image[image_mask==False]
                inpaint_feature = inpaint_feature.reshape((self.num_inpaints, -1, 256))
                image_feature = image_feature.reshape((1, -1, 256)) 
                feature = np.concatenate((inpaint_feature, image_feature), axis=0)
                self.features.append(torch.tensor(feature))

                image = dataset_non_train["train"][dataset_non_train_permutations[i].item()]["image"]
                audio = self.mel.image_to_audio(image)
                image = np.array(image)

                mask = np.ones((audio.shape[0]), dtype=bool)
                top_k = top_k_windows(audio, self.window_size, entropy, self.num_gaps)
                for tk in top_k:
                    mask[tk[1]:tk[1]+self.window_size] = False
                image_mask = mask[::510]
                image_mask = image_mask.reshape(1, 1, 256)
                
                masked_audio = audio * mask
                
                inpaint_image = self.audio_diffusion.generate_spectrogram_and_audio_from_audio(
                    raw_audio=masked_audio,
                    batch_size=self.num_inpaints,
                    start_step=900,
                    step_generator=torch.Generator(device="cuda"))

                image = image.reshape((1, 256, 256)) 
                image_mask = np.tile(image_mask, (inpaint_image.shape[0], inpaint_image.shape[1], 1))
                inpaint_feature = inpaint_image[image_mask==False]
                image_feature = image[image_mask==False]
                inpaint_feature = inpaint_feature.reshape((self.num_inpaints, -1, 256))
                image_feature = image_feature.reshape((1, -1, 256)) 
                feature = np.concatenate((inpaint_feature, image_feature), axis=0)
                self.features.append(torch.tensor(feature))

            self.labels = torch.tensor([1, 0]*dataset_config["num_train_examples"])
            if dataset_config["save_dir"] is not None:
                dataset = {
                    "features": self.features,
                    "labels": self.labels
                }
                torch.save(dataset, dataset_config["save_dir"])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx].type(torch.float32)/255, self.labels[idx].type(torch.float32)