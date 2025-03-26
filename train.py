from datasets import download_and_extract_speech_commands_dataset, SpeechCommandsDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import check_model, check_forward_pass, count_precise_macs
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='./config', config_name='phi_gru')
def train(cfg: DictConfig):

    experiment_name = cfg.experiment_name
    print(f"Experiment: {experiment_name}")

    # Download and extract the Speech Commands dataset
    download_and_extract_speech_commands_dataset()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define allowed classes
    # You need to define ALLOWED_CLASSES based on the classes you want to use
    ALLOWED_CLASSES = cfg.dataset.allowed_classes
    NUM_CLASSES = len(ALLOWED_CLASSES)

    # Preload dataset, set to False for faster tests
    preload = cfg.dataset.preload

    # Create datasets for each split SpeechCommandsDataset
    train_dataset = instantiate(cfg.dataset.train, preload=preload)
    val_dataset = instantiate(cfg.dataset.val, preload=preload)
    test_dataset = instantiate(cfg.dataset.test, preload=preload)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model, loss, optimizer
    model = instantiate(cfg.model, num_classes=NUM_CLASSES).to(device)

    # Check model
    check_model(model)
    check_forward_pass(model, train_dataset, device)

    # MACC Computation
    with torch.no_grad():
        total_test_macs = 0
        test_total = 0  # Ensure this is initialized before the loop
        
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Calculate MACs for this batch
            batch_macs = count_precise_macs(model, inputs)
            total_test_macs += batch_macs
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)  # Accumulate total number of samples
            
            # (rest of the existing testing code remains the same)
        
        # Print total MACs and MACs per sample
        print(f"\nTotal Test MACs: {total_test_macs:,}")
        print(f"Average MACs per Sample: {total_test_macs / test_total:,.2f}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    num_epochs = 200

    # Learning rate scheduler
    scheduler = instantiate(cfg.scheduler, optimizer)

    # Training loop
    best_val_acc = 0.0
    best_model_path = f"best_{experiment_name}.pth"
    
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
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")  # Print current learning rate
        print("-" * 40)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val acc: {val_acc:.2f}%")
    
    print(f"BEST VAL ACC: {best_val_acc:.2f}%")
    
    # Testing with the best model
    print("\nEvaluating on test set...")
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
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(ALLOWED_CLASSES):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{class_name}: {class_acc:.2f}%")
    
    # Print confusion matrix (optional)
    # You may want to save this to a file or visualize it
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
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
        print("Confusion matrix visualization saved to confusion_matrix.png")
    except ImportError:
        print("Matplotlib or seaborn not available for plotting confusion matrix")


if __name__ == "__main__":
    train()