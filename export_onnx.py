import torch
import torch.onnx
# from models import Improved__FC_GRU_ATT
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path='./config', config_name='phi_gru')
def main(cfg: DictConfig):
    # Instantiate your model
    model = instantiate(cfg.model, num_classes=35)
    model.eval()
    print(model)
    # Create a dummy input - use the appropriate input shape for your model
    # For audio, this might be [batch_size, waveform_length] 
    # Example: [1, 16000] for 1 second of audio at 16kHz
    dummy_input = torch.randn(1, 16000)

    # Export the model
    torch.onnx.export(
        model,                       # model being run
        dummy_input,                 # model input (or a tuple for multiple inputs)
        "improved_phi_fc_model.onnx",  # where to save the model
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=19,            # the ONNX version to export the model to
        do_constant_folding=True,    # whether to execute constant folding for optimization
        input_names=['input'],       # the model's input names
        output_names=['output'],     # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},  # variable length axes
            'output': {0: 'batch_size'}
        }
    )

if __name__ == "__main__":
    main()