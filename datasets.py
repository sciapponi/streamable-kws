import os
import requests
import tarfile
import random
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio.sox_effects as sox_effects
import torch

# ALLOWED_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def download_and_extract_speech_commands_dataset():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    save_path = "speech_commands.tar.gz"
    extract_path = "speech_commands_dataset"

    if not os.path.exists(extract_path):
        print("Downloading Speech Commands dataset...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print("Extracting dataset...")
        with tarfile.open(save_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        os.remove(save_path)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, transform=None, allowed_classes=["up", "down", "left", "right", "nothing", "other"],
                 subset="training", augment=False, preload=False):
        """
        Speech Commands Dataset with support for 'nothing' and 'other' classes

        Args:
            root_dir (string): Directory with the Speech Commands dataset
            transform (callable, optional): Optional transform to be applied on a sample
            allowed_classes (list): List of classes to include, should contain "nothing" and "other" for those classes
            subset (string): Which subset to use ('training', 'validation', 'testing')
            augment (bool): Whether to apply augmentation to the data
            preload (bool): If True, preload all audio files into memory
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.augment = augment and subset == "training"  # Only augment training data
        self.preload = preload

        # Make a copy of allowed_classes to avoid modifying the input
        self.allowed_classes = list(allowed_classes)

        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.allowed_classes)}

        self.file_list = []
        self.labels = []
        self.preloaded_data = {}

        # Get the test and validation file lists
        test_files = set()
        validation_files = set()

        test_list_path = os.path.join(root_dir, "testing_list.txt")
        if os.path.exists(test_list_path):
            with open(test_list_path, 'r') as f:
                for line in f:
                    test_files.add(line.strip())

        val_list_path = os.path.join(root_dir, "validation_list.txt")
        if os.path.exists(val_list_path):
            with open(val_list_path, 'r') as f:
                for line in f:
                    validation_files.add(line.strip())

        # List all command word files (up, down, left, right)
        standard_classes = [cls for cls in self.allowed_classes if cls not in ["nothing", "other"]]
        for class_name in standard_classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.wav'):
                        relative_path = os.path.join(class_name, file)
                        full_path = os.path.join(root_dir, relative_path)

                        # Check which subset this file belongs to
                        in_test = relative_path in test_files
                        in_val = relative_path in validation_files

                        should_include = False
                        if self.subset == "testing" and in_test:
                            should_include = True
                        elif self.subset == "validation" and in_val:
                            should_include = True
                        elif self.subset == "training" and not in_test and not in_val:
                            should_include = True

                        if should_include:
                            self.file_list.append(full_path)
                            self.labels.append(self.class_to_idx[class_name])

        # Add "nothing" class if requested
        if "nothing" in self.allowed_classes:
            self.add_nothing_class()

        # Add "other" class if requested
        if "other" in self.allowed_classes:
            self.add_other_class()

        # Preload data to speed up training
        if self.preload:
            self.preloaded_data = {path: self.load_audio(path) for path in self.file_list}

    def add_nothing_class(self):
        """Add background noise samples as the 'nothing' class."""
        background_dir = os.path.join(self.root_dir, "_background_noise_")

        # Calculate target number of "nothing" samples
        # Aim to make "nothing" class roughly the same size as the standard classes
        standard_classes = [cls for cls in self.allowed_classes if cls not in ["nothing", "other"]]
        target_count = 0
        for class_name in standard_classes:
            class_count = sum(1 for label in self.labels if label == self.class_to_idx[class_name])
            target_count = max(target_count, class_count)

        if os.path.isdir(background_dir):
            bg_files = [f for f in os.listdir(background_dir) if f.endswith('.wav')]

            if not bg_files:
                print(f"Warning: No background noise files found in {background_dir}")
                return

            # Determine how many segments to create per file
            segments_per_file = max(10, (target_count // len(bg_files)) + 1)

            # Create segments from background noise files
            for file in bg_files:
                bg_path = os.path.join(background_dir, file)

                # For background noise, we'll create multiple segments from each file
                if self.subset == "training":
                    num_segments = segments_per_file
                else:
                    num_segments = segments_per_file // 2  # Fewer for validation/testing

                for i in range(num_segments):
                    self.file_list.append(bg_path + f"#{i}")  # Use # to mark it as a segment
                    self.labels.append(self.class_to_idx["nothing"])

        print(f"Added {sum(1 for label in self.labels if label == self.class_to_idx['nothing'])} 'nothing' samples for {self.subset} set")

    def add_other_class(self):
        """Add samples from non-target words as the 'other' class."""
        # Get all subdirectories in the root directory
        standard_classes = [cls for cls in self.allowed_classes if cls not in ["nothing", "other"]]
        all_classes = [d for d in os.listdir(self.root_dir)
                      if os.path.isdir(os.path.join(self.root_dir, d))
                      and d not in standard_classes
                      and not d.startswith('_')]  # Skip special directories like _background_noise_

        # Get test and validation files
        test_files = set()
        validation_files = set()

        test_list_path = os.path.join(self.root_dir, "testing_list.txt")
        if os.path.exists(test_list_path):
            with open(test_list_path, 'r') as f:
                for line in f:
                    test_files.add(line.strip())

        val_list_path = os.path.join(self.root_dir, "validation_list.txt")
        if os.path.exists(val_list_path):
            with open(val_list_path, 'r') as f:
                for line in f:
                    validation_files.add(line.strip())

        # Calculate target number of "other" samples per subset
        # Aim to make "other" class roughly the same size as the standard classes
        target_count = 0
        for class_name in standard_classes:
            class_count = sum(1 for label in self.labels if label == self.class_to_idx[class_name])
            target_count = max(target_count, class_count)

        # For each non-target class, add samples as "other"
        samples_added = 0

        # First pass: add a baseline number from each class
        samples_per_class = 25 if self.subset == "training" else 10

        for class_name in all_classes:
            class_dir = os.path.join(self.root_dir, class_name)
            files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

            # Add more samples from each class to increase diversity
            num_samples = min(samples_per_class, len(files))
            selected_files = random.sample(files, num_samples)

            for file in selected_files:
                relative_path = os.path.join(class_name, file)
                full_path = os.path.join(self.root_dir, relative_path)

                # Check which subset this file belongs to
                in_test = relative_path in test_files
                in_val = relative_path in validation_files

                should_include = False
                if self.subset == "testing" and in_test:
                    should_include = True
                elif self.subset == "validation" and in_val:
                    should_include = True
                elif self.subset == "training" and not in_test and not in_val:
                    should_include = True

                if should_include:
                    self.file_list.append(full_path)
                    self.labels.append(self.class_to_idx["other"])
                    samples_added += 1

        # Second pass: add more files if needed to reach target count
        if samples_added < target_count * 0.9:  # If we're below 90% of target
            # Flatten the list of all available files
            all_other_files = []
            for class_name in all_classes:
                class_dir = os.path.join(self.root_dir, class_name)
                for file in os.listdir(class_dir):
                    if file.endswith('.wav'):
                        relative_path = os.path.join(class_name, file)
                        full_path = os.path.join(self.root_dir, relative_path)

                        # Check which subset this file belongs to
                        in_test = relative_path in test_files
                        in_val = relative_path in validation_files

                        should_include = False
                        if self.subset == "testing" and in_test:
                            should_include = True
                        elif self.subset == "validation" and in_val:
                            should_include = True
                        elif self.subset == "training" and not in_test and not in_val:
                            should_include = True

                        if should_include and full_path not in self.file_list:
                            all_other_files.append((full_path, relative_path))

            # Add more files randomly until we reach the target
            if all_other_files:
                random.shuffle(all_other_files)
                additional_needed = int(target_count - samples_added)
                additional_files = all_other_files[:additional_needed]

                for full_path, _ in additional_files:
                    self.file_list.append(full_path)
                    self.labels.append(self.class_to_idx["other"])

        print(f"Added {sum(1 for label in self.labels if label == self.class_to_idx['other'])} 'other' samples for {self.subset} set")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        if self.preload:
            waveform = self.preloaded_data[audio_path]
        else:
            waveform = self.load_audio(audio_path)

        # Apply augmentation if enabled
        if self.augment:
            waveform = self.apply_augmentation(waveform)

        # Apply additional transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

    def load_audio(self, audio_path):
        # Check if this is a background noise segment
        if "#" in audio_path:
            base_path, segment_id = audio_path.split("#")
            segment_id = int(segment_id)

            # Load the full background noise file
            waveform, sample_rate = torchaudio.load(base_path)

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Select a random 1-second segment
            total_samples = waveform.shape[1]
            if total_samples <= 16000:
                start_idx = 0
            else:
                # Use segment_id to deterministically select different segments
                max_start = total_samples - 16000
                start_idx = (segment_id * 1234) % max_start  # Pseudo-random but deterministic

            segment = waveform[:, start_idx:start_idx + 16000]

            # Ensure the segment is exactly 16000 samples
            segment = self.ensure_length(segment)

            # Normalize
            if segment.abs().max() > 0:
                segment = segment / segment.abs().max()

            return segment

        # Regular audio files
        waveform, sample_rate = torchaudio.load(audio_path)  # Load waveform

        # Convert stereo to MONO (if needed)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert stereo to mono

        # Ensure waveform is exactly 16000 samples
        num_samples = waveform.shape[1]
        if num_samples < 16000:
            waveform = F.pad(waveform, (0, 16000 - num_samples))  # Pad with zeros
        elif num_samples > 16000:
            waveform = waveform[:, :16000]  # Trim to 16000 samples

        # Normalize audio (avoid division by zero)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        return waveform

    def apply_augmentation(self, waveform):
        aug_transforms = [
            self.add_noise,
            self.time_mask,
            self.freq_mask,
            self.time_shift,  # Added time shift as an augmentation option
        ]
        random.shuffle(aug_transforms)  # Apply augmentations in random order
        for aug in aug_transforms:
            if random.random() < 0.2:  # 20% chance to apply each augmentation
                waveform = aug(waveform)

        # Ensure the waveform is exactly 16000 samples after augmentation
        waveform = self.ensure_length(waveform)

        return waveform

    def ensure_length(self, waveform, target_length=16000):
        num_samples = waveform.shape[1]

        # Pad if shorter than the target length
        if num_samples < target_length:
            waveform = F.pad(waveform, (0, target_length - num_samples))  # Pad with zeros
        # Trim if longer than the target length
        elif num_samples > target_length:
            waveform = waveform[:, :target_length]

        return waveform

    def add_noise(self, waveform, noise_level=0.0001):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def time_mask(self, waveform, time_mask_param=9):
        transform = T.TimeMasking(time_mask_param=time_mask_param)
        return transform(waveform)

    def freq_mask(self, waveform, freq_mask_param=5):
        transform = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        return transform(waveform)

    def time_shift(self, waveform, shift_limit=0.2):
        shift = int(shift_limit * 16000 * (random.random() - 0.5))
        return torch.roll(waveform, shifts=shift, dims=1)

    def speed_perturb(self, waveform, rate_min=0.9, rate_max=1.1):
        rate = random.uniform(rate_min, rate_max)
        transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=int(16000 * rate))
        return transform(waveform)

    def reverb(self, waveform, sample_rate=16000, reverberance=50):
        """Apply SoX-based reverberation effect."""
        effects = [
            ["reverb", str(reverberance)]  # Apply reverb with given strength (0-100)
        ]
        waveform, _ = sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        return waveform

# class SpeechCommandsDataset(Dataset):
#     def __init__(self, root_dir, transform=None, allowed_classes=[], subset="training", augment=False, preload=False,
#                  return_spectrogram=True, n_mel_bins=40, n_fft=400, hop_length=160, win_length=400):
#         """
#         Speech Commands Dataset with Spectrogram Support

#         Args:
#             root_dir (string): Directory with the Speech Commands dataset
#             transform (callable, optional): Optional transform to be applied on a sample
#             allowed_classes (list): List of classes to include in the dataset
#             subset (string): Which subset to use ('training', 'validation', 'testing')
#             augment (bool): Whether to apply augmentation to the data
#             preload (bool): If True, preload all audio files into memory
#             return_spectrogram (bool): If True, return mel spectrogram instead of waveform
#             n_mel_bins (int): Number of mel bins for spectrogram
#             n_fft (int): FFT size for spectrogram
#             hop_length (int): Hop length for spectrogram
#             win_length (int): Window length for spectrogram
#         """
#         self.allowed_classes = allowed_classes
#         self.root_dir = root_dir
#         self.transform = transform
#         self.subset = subset
#         self.augment = augment and subset == "training"  # Only augment training data
#         self.preload = preload
#         self.return_spectrogram = return_spectrogram
#         self.file_list = []
#         self.labels = []
#         self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.allowed_classes)}
#         self.preloaded_data = {}

#         # Spectrogram parameters
#         self.n_mel_bins = n_mel_bins
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.sample_rate = 16000

#         # Initialize mel spectrogram transform
#         self.mel_spectrogram = T.MelSpectrogram(
#             sample_rate=self.sample_rate,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             win_length=self.win_length,
#             n_mels=self.n_mel_bins,
#             center=True,
#             pad_mode="reflect",
#             power=2.0,
#         )

#         # Get the test and validation file lists
#         test_files = set()
#         validation_files = set()

#         test_list_path = os.path.join(root_dir, "testing_list.txt")
#         if os.path.exists(test_list_path):
#             with open(test_list_path, 'r') as f:
#                 for line in f:
#                     test_files.add(line.strip())

#         val_list_path = os.path.join(root_dir, "validation_list.txt")
#         if os.path.exists(val_list_path):
#             with open(val_list_path, 'r') as f:
#                 for line in f:
#                     validation_files.add(line.strip())

#         # List all files based on the subset
#         for class_name in self.allowed_classes:
#             class_dir = os.path.join(root_dir, class_name)
#             if os.path.isdir(class_dir):
#                 for file in os.listdir(class_dir):
#                     if file.endswith('.wav'):
#                         relative_path = os.path.join(class_name, file)  # Path as it appears in list files
#                         full_path = os.path.join(root_dir, relative_path)

#                         # Check which subset this file belongs to
#                         in_test = relative_path in test_files
#                         in_val = relative_path in validation_files

#                         should_include = False
#                         if self.subset == "testing" and in_test:
#                             should_include = True
#                         elif self.subset == "validation" and in_val:
#                             should_include = True
#                         elif self.subset == "training" and not in_test and not in_val:
#                             should_include = True

#                         if should_include:
#                             self.file_list.append(full_path)
#                             self.labels.append(self.class_to_idx[class_name])

#         # Preload data to speed up training
#         if self.preload:
#             if self.return_spectrogram:
#                 self.preloaded_data = {path: self.waveform_to_spectrogram(self.load_audio(path))
#                                       for path in self.file_list}
#             else:
#                 self.preloaded_data = {path: self.load_audio(path) for path in self.file_list}

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         audio_path = self.file_list[idx]
#         label = self.labels[idx]

#         if self.preload:
#             if self.return_spectrogram:
#                 feature = self.preloaded_data[audio_path]
#             else:
#                 waveform = self.preloaded_data[audio_path]
#         else:
#             waveform = self.load_audio(audio_path)

#             # Apply augmentation if enabled (only for waveform)
#             if self.augment:
#                 waveform = self.apply_augmentation(waveform)

#             # Convert to spectrogram if needed
#             if self.return_spectrogram:
#                 feature = self.waveform_to_spectrogram(waveform)
#             else:
#                 feature = waveform

#         # Apply additional transform if provided
#         if self.transform:
#             feature = self.transform(feature)

#         return feature, label

#     def waveform_to_spectrogram(self, waveform):
#         """Convert waveform to mel spectrogram"""
#         with torch.no_grad():
#             mel_spec = self.mel_spectrogram(waveform)
#             # Convert to log scale (dB) - prevent numerical instability
#             eps = 1e-10
#             mel_spec = torch.log(mel_spec + eps)
#         return mel_spec

#     def load_audio(self, audio_path):
#         waveform, sample_rate = torchaudio.load(audio_path)  # Load waveform

#         # Convert stereo to MONO (if needed)
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)  # Convert stereo to mono

#         # Ensure waveform is exactly 16000 samples
#         num_samples = waveform.shape[1]
#         if num_samples < 16000:
#             waveform = F.pad(waveform, (0, 16000 - num_samples))  # Pad with zeros
#         elif num_samples > 16000:
#             waveform = waveform[:, :16000]  # Trim to 16000 samples

#         # Normalize audio (avoid division by zero)
#         if waveform.abs().max() > 0:
#             waveform = waveform / waveform.abs().max()

#         return waveform

#     def apply_augmentation(self, waveform):
#         """Apply time-domain augmentations to waveform"""
#         aug_transforms = [
#             self.add_noise,
#             self.time_mask,
#             self.time_shift,
#             self.speed_perturb
#         ]
#         random.shuffle(aug_transforms)  # Apply augmentations in random order
#         for aug in aug_transforms:
#             if random.random() < 0.2:  # 20% chance to apply each augmentation
#                 waveform = aug(waveform)

#         # Ensure the waveform is exactly 16000 samples after augmentation
#         waveform = self.ensure_length(waveform)

#         return waveform

#     def apply_spec_augmentation(self, spectrogram):
#         """Apply augmentations directly to spectrogram"""
#         if random.random() < 0.2:
#             spectrogram = self.freq_mask(spectrogram)
#         if random.random() < 0.2:
#             spectrogram = self.time_mask(spectrogram)
#         return spectrogram

#     def ensure_length(self, waveform, target_length=16000):
#         num_samples = waveform.shape[1]

#         # Pad if shorter than the target length
#         if num_samples < target_length:
#             waveform = F.pad(waveform, (0, target_length - num_samples))  # Pad with zeros
#         # Trim if longer than the target length
#         elif num_samples > target_length:
#             waveform = waveform[:, :target_length]

#         return waveform

#     def add_noise(self, waveform, noise_level=0.005):
#         noise = torch.randn_like(waveform) * noise_level
#         return waveform + noise

#     def time_mask(self, waveform_or_spec, mask_param=10):
#         """Apply time masking to either waveform or spectrogram"""
#         if waveform_or_spec.dim() == 2 and waveform_or_spec.shape[0] == 1:
#             # It's a waveform, shape: [1, time]
#             time_mask_param = mask_param * 100  # Different scale for waveform
#             transform = T.TimeMasking(time_mask_param=time_mask_param)
#         else:
#             # It's a spectrogram, shape: [1, freq, time]
#             transform = T.TimeMasking(time_mask_param=mask_param)
#         return transform(waveform_or_spec)

#     def freq_mask(self, spectrogram, freq_mask_param=8):
#         """Apply frequency masking to spectrogram"""
#         transform = T.FrequencyMasking(freq_mask_param=freq_mask_param)
#         return transform(spectrogram)

#     def time_shift(self, waveform, shift_limit=0.2):
#         shift = int(shift_limit * 16000 * (random.random() - 0.5))
#         return torch.roll(waveform, shifts=shift, dims=1)

#     def speed_perturb(self, waveform, rate_min=0.9, rate_max=1.1):
#         rate = random.uniform(rate_min, rate_max)
#         transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=int(16000 * rate))
#         waveform = transform(waveform)
#         # Ensure length is correct after speed perturbation
#         return self.ensure_length(waveform)

#     def reverb(self, waveform, sample_rate=16000, reverberance=50):
#         """Apply SoX-based reverberation effect."""
#         try:
#             effects = [
#                 ["reverb", str(reverberance)]  # Apply reverb with given strength (0-100)
#             ]
#             waveform, _ = sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
#             return waveform
#         except:
#             # In case of an error with sox_effects, return the original waveform
#             return waveform

if __name__==  "__main__":
    download_and_extract_speech_commands_dataset()
    test_dataset = SpeechCommandsDataset(root_dir="speech_commands_dataset", subset="testing", preload=False)
    print(len(test_dataset))
