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

    log.info(f"Training samples: {len(train_dataset)}")
    log.info(f"Validation samples: {len(val_dataset)}")
    log.info(f"Testing samples: {len(test_dataset)}")

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

    # MACC Computation
    with torch.no_grad():
        total_test_macs = 0
        test_total = 0

        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            batch_macs = count_precise_macs(model, inputs)
            total_test_macs += batch_macs

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)

        log.info(f"Total Test MACs: {total_test_macs:,}")
        log.info(f"Average MACs per Sample: {total_test_macs / test_total:,.2f}")

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss() # removed label smoothing
    # criterion= FocalLoss()
    optimizer = instantiate(cfg.optimizer, model.parameters())
    num_epochs = cfg.training.epochs

    # Learning rate scheduler
    scheduler = instantiate(cfg.scheduler, optimizer)

    # Training loop with early stopping
    best_val_acc = 0.0
    best_model_path = f"{output_dir}/best_{experiment_name}.pth"

    # Early stopping parameters
    patience = cfg.training.get('patience', num_epochs)  # Number of epochs to wait before stopping
    min_delta = cfg.training.get('min_delta', 0.0005)  # Minimum change to qualify as improvement
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

            optimizer.zero_grad()
            outputs = model(inputs)
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

                outputs = model(inputs)
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

            outputs = model(inputs)
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

    # ONNX Export
    log.info("Exporting model to ONNX...")

    # Re-instantiate model in export mode
    export_model = instantiate(cfg.model, num_classes=NUM_CLASSES, export_mode=True).to(device)
    export_model.load_state_dict(torch.load(best_model_path))
    export_model.eval()

    # Get a waveform sample and convert it to a spectrogram
    # This should match the input your model expects when export_mode=True
    sample_waveform, sr = test_dataset[0]  # raw waveform
    sample_waveform = sample_waveform.unsqueeze(0).to(device)  # Add batch dim

    # Transform waveform to spectrogram like your preprocessing pipeline
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
    mel_transform = MelSpectrogram(
        sample_rate=sr, n_fft=cfg.model.n_fft, hop_length=cfg.model.hop_length, n_mels=cfg.model.n_mel_bins
    ).to(device)
    db_transform = AmplitudeToDB().to(device)

    spectrogram = db_transform(mel_transform(sample_waveform))  # shape: [1, 40, time]
    spectrogram = spectrogram.unsqueeze(1)  # add channel dim -> [1, 1, 40, time]
    spectrogram = spectrogram.squeeze(1)  # model expects [B, 40, time]

    # Export model to ONNX with dynamic time axis
    export_path = f"{output_dir}/{experiment_name}_export.onnx"
    torch.onnx.export(
        export_model,
        spectrogram,  # input tensor
        export_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'time'},  # dynamic time dimension
            'output': {1: 'class'}  # optional
        }
    )
    log.info(f"ONNX model exported to: {export_path}")



if __name__ == "__main__":
    train()
