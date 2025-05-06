import os
import torch
import torchaudio
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import hydra
from datasets import SpeechCommandsDataset
from models import Improved_Phi_GRU_ATT


class SpectrogramTransform:
    def __init__(self, n_fft=400, hop_length=160, n_mels=40):
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        # Compute mel spectrogram
        spec = self.mel_spec(waveform)
        # Convert to dB scale
        spec = self.amplitude_to_db(spec)
        return spec

class ModelTester:
    def __init__(self,
                 onnx_model_path,
                 dataset_root,
                 class_names,
                 conf,
                 batch_size=32):

        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.class_names = class_names

        # Load ONNX model only
        self.onnx_session = self.load_onnx_model(onnx_model_path)

        # Setup transforms for spectrogram computation
        self.spec_transform = SpectrogramTransform()

        # Create dataloader
        _, self.test_loader_specs = self.create_dataloaders(
            dataset_root, class_names, batch_size)

    def load_onnx_model(self, path):
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider'] + providers
        return ort.InferenceSession(path, providers=providers)

    def create_dataloaders(self, dataset_root, class_names, batch_size):
        # Only return ONNX dataset loader
        test_dataset_specs = SpeechCommandsDataset(
            root_dir=dataset_root,
            transform=self.spec_transform,
            allowed_classes=class_names,
            subset="testing",
            augment=False
        )

        test_loader_specs = DataLoader(
            test_dataset_specs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        return None, test_loader_specs

    def test_onnx_model(self):
        all_preds = []
        all_labels = []

        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name

        for specs, labels in tqdm(self.test_loader_specs, desc="Testing ONNX model"):
            input_data = specs.numpy().astype(np.float32)
            mean = np.mean(input_data, axis=(2, 3), keepdims=True)
            std = np.std(input_data, axis=(2, 3), keepdims=True) + 1e-5
            input_data = (input_data - mean) / std

            outputs = self.onnx_session.run([output_name], {input_name: input_data})
            preds = np.argmax(outputs[0], axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        return np.array(all_preds), np.array(all_labels)

    def run_test(self):
        print("Testing ONNX model only...")
        onnx_preds, onnx_labels = self.test_onnx_model()

        accuracy = np.mean(onnx_preds == onnx_labels)
        print(f"ONNX model accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(onnx_labels, onnx_preds, target_names=self.class_names))

        # Plot confusion matrix
        cm = confusion_matrix(onnx_labels, onnx_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - ONNX Model')
        plt.tight_layout()
        plt.savefig('confusion_matrix_ONNX_Model.png')
        plt.close()


@hydra.main(version_base=None, config_path='/home/ste/Code/streamable-kws/logs/4_classes_hd64/2025-05-05_15-22-03/.hydra', config_name='config')
def main(conf):
    onnx_model_path = "/home/ste/Code/streamable-kws/logs/4_classes_hd64/2025-05-05_15-22-03/model_opt.onnx"
    dataset_root = "/home/ste/Code/streamable-kws/speech_commands_dataset"
    class_names = ["up", "down", "left", "right"]

    tester = ModelTester(
        onnx_model_path=onnx_model_path,
        dataset_root=dataset_root,
        class_names=class_names,
        conf=conf,
        batch_size=1
    )

    tester.run_test()



if __name__ == "__main__":
    main()
