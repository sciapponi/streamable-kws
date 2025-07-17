from datasets import download_and_extract_speech_commands_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils import check_model, check_forward_pass, count_precise_macs
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import logging
from losses import FocalLoss

class ChunkedDataset(torch.utils.data.Dataset):
    """Wrapper dataset that splits audio into smaller chunks"""
    def __init__(self, original_dataset, chunk_size_ms=200, sample_rate=16000):
        self.original_dataset = original_dataset
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)
        self.hop_size = int(sample_rate * 20 / 1000)  # 20ms hop between chunks

    def __len__(self):
        # Each 1-second audio produces multiple chunks
        original_len = len(self.original_dataset)
        audio_length = 1 * self.sample_rate  # 1 second
        chunks_per_audio = (audio_length - self.chunk_size) // self.hop_size + 1
        return original_len * chunks_per_audio

    def __getitem__(self, idx):
        audio_idx = idx // ((1 * self.sample_rate - self.chunk_size) // self.hop_size + 1)
        chunk_idx = idx % ((1 * self.sample_rate - self.chunk_size) // self.hop_size + 1)

        audio, label = self.original_dataset[audio_idx]
        if audio.dim() > 1:
            audio = audio.squeeze(0)  # remove channel dimension if present

        if len(audio) < self.sample_rate:
            padding = self.sample_rate - len(audio)
            audio = torch.cat([audio, torch.zeros(padding)], dim=0)


        # Safety check: trim or pad audio to expected 1 second
        if len(audio) < self.sample_rate:
            audio = torch.cat([audio, torch.zeros(self.sample_rate - len(audio))])
        elif len(audio) > self.sample_rate:
            audio = audio[:self.sample_rate]

        start = chunk_idx * self.hop_size
        end = start + self.chunk_size
        chunk = audio[start:end]

        # Pad if necessary
        if len(chunk) < self.chunk_size:
            padding = self.chunk_size - len(chunk)
            chunk = torch.cat([chunk, torch.zeros(padding)])

        return chunk, label



@hydra.main(version_base=None, config_path='./config', config_name='phi_gru')
def train(cfg: DictConfig):
    # Some Hydra Configurations things
    log = logging.getLogger(__name__)
    experiment_name = cfg.experiment_name
    log.info(f"Experiment: {experiment_name}")
    output_dir = HydraConfig.get().runtime.output_dir

    # Download and extract the Speech Commands dataset
    download_and_extract_speech_commands_dataset()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Define allowed classes
    ALLOWED_CLASSES = cfg.dataset.allowed_classes
    NUM_CLASSES = len(ALLOWED_CLASSES)

    # Preload dataset, set to False for faster tests
    preload = cfg.dataset.preload

    # Create datasets for each split SpeechCommandsDataset
    train_dataset = instantiate(cfg.dataset.train, preload=preload, allowed_classes=ALLOWED_CLASSES)
    val_dataset = instantiate(cfg.dataset.val, preload=preload, allowed_classes=ALLOWED_CLASSES)
    test_dataset = instantiate(cfg.dataset.test, preload=preload, allowed_classes=ALLOWED_CLASSES)

    # Wrap datasets with chunking
    chunk_size_ms = cfg.training.get('chunk_size_ms', 200)  # Default 200ms chunks
    train_dataset = ChunkedDataset(train_dataset, chunk_size_ms=chunk_size_ms)
    val_dataset = ChunkedDataset(val_dataset, chunk_size_ms=chunk_size_ms)
    test_dataset = ChunkedDataset(test_dataset, chunk_size_ms=chunk_size_ms)

    log.info(f"Training samples: {len(train_dataset)} (chunked)")
    log.info(f"Validation samples: {len(val_dataset)} (chunked)")
    log.info(f"Testing samples: {len(test_dataset)} (chunked)")

    # DataLoaders
    batch_size = cfg.training.get('batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = instantiate(cfg.model, num_classes=NUM_CLASSES).to(device)

    # Check model
    check_model(model)
    check_forward_pass(model, train_dataset, device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, model.parameters())
    num_epochs = cfg.training.epochs

    # Learning rate scheduler
    scheduler = instantiate(cfg.scheduler, optimizer)

    # Training loop with early stopping
    best_val_acc = 0.0
    best_model_path = f"{output_dir}/best_{experiment_name}.pth"

    # Early stopping parameters
    patience = cfg.training.get('patience', num_epochs)
    min_delta = cfg.training.get('min_delta', 0.0005)
    early_stop_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Initialize hidden state for each batch
            hidden_state = None

            optimizer.zero_grad()

            # Forward pass through all chunks in the batch
            outputs, _ = model(inputs, hidden_state)

            # Calculate loss - we use the same label for all chunks from the same original audio
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Initialize hidden state for each validation batch
                hidden_state = None

                outputs, _ = model(inputs, hidden_state)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Step the scheduler based on validation loss
        scheduler.step()

        log.info(f"Epoch {epoch+1}/{num_epochs}")
        log.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        log.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        log.info(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        log.info("-" * 40)

        # Check for improvement in validation loss for early stopping
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            log.info(f"No improvement in validation loss for {early_stop_counter}/{patience} epochs")

            if early_stop_counter >= patience:
                log.info(f"Early stopping triggered after {epoch+1} epochs!")
                break

        # Save best model (based on accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            log.info(f"New best model saved with val acc: {val_acc:.2f}%")

    log.info(f"BEST VAL ACC: {best_val_acc:.2f}%")

    # Testing with the best model
    log.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_correct = 0
    test_total = 0
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Initialize hidden state for each test batch
            hidden_state = None

            outputs, _ = model(inputs, hidden_state)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                class_correct[label] += (pred == label).item()
                class_total[label] += 1
                confusion_matrix[label, pred] += 1

    # Calculate overall test accuracy
    test_acc = 100 * test_correct / test_total
    log.info(f"Test Accuracy: {test_acc:.2f}%")

    # log.info per-class accuracy
    log.info("\nPer-class accuracy:")
    for i, class_name in enumerate(ALLOWED_CLASSES):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            log.info(f"{class_name}: {class_acc:.2f}%")

    # log.info confusion matrix
    log.info("\nConfusion Matrix:")
    log.info(confusion_matrix)

    # Save confusion matrix to a CSV file
    import pandas as pd
    confusion_df = pd.DataFrame(confusion_matrix.cpu().numpy(),
                               index=ALLOWED_CLASSES,
                               columns=ALLOWED_CLASSES)
    confusion_df.to_csv(f"{experiment_name}confusion_matrix.csv")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_df, annot=False, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        log.info("Confusion matrix visualization saved to confusion_matrix.png")
    except ImportError:
        log.info("Matplotlib or seaborn not available for plotting confusion matrix")

    # ONNX Export (unchanged from original)
    log.info("Exporting model to ONNX...")
    export_model = instantiate(cfg.model, num_classes=NUM_CLASSES, export_mode=True).to(device)
    state_dict = torch.load(best_model_path)
    missing_keys, unexpected_keys = export_model.load_state_dict(state_dict, strict=False)
    log.warning(f"Missing keys: {missing_keys}")
    log.warning(f"Unexpected keys: {unexpected_keys}")
    export_model.eval()

    export_path = f"{output_dir}/{experiment_name}_export.onnx"

    if hasattr(export_model, 'export_onnx') and callable(export_model.export_onnx):
        log.info("Using model's built-in export_onnx method...")
        sample_waveform, sr = test_dataset.original_dataset[0]
        sample_waveform = sample_waveform.unsqueeze(0).to(device)
        from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
        mel_transform = MelSpectrogram(
            sample_rate=sr, n_fft=cfg.model.n_fft, hop_length=cfg.model.hop_length, n_mels=cfg.model.n_mel_bins
        ).to(device)
        db_transform = AmplitudeToDB().to(device)
        spectrogram = db_transform(mel_transform(sample_waveform))

        batch_size = 1
        n_mels = cfg.model.n_mel_bins
        time_steps = spectrogram.size(2)
        input_shape = (batch_size, n_mels, time_steps)

        export_model.export_onnx(save_path=export_path, input_shape=input_shape)
        log.info(f"Model-specific export completed to: {export_path}")
    else:
        log.info("Performing generic ONNX export...")
        sample_waveform, sr = test_dataset.original_dataset[0]
        sample_waveform = sample_waveform.unsqueeze(0).to(device)
        from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
        mel_transform = MelSpectrogram(
            sample_rate=sr, n_fft=cfg.model.n_fft, hop_length=cfg.model.hop_length, n_mels=cfg.model.n_mel_bins
        ).to(device)
        db_transform = AmplitudeToDB().to(device)
        spectrogram = db_transform(mel_transform(sample_waveform))

        torch.onnx.export(
            export_model,
            spectrogram,
            export_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {2: 'time'},
                'output': {1: 'class'}
            }
        )

    log.info(f"ONNX model exported to: {export_path}")


if __name__ == "__main__":
    train()
