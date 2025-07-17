import numpy as np
import onnxruntime as ort
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchaudio
import argparse


def test_onnx_streaming(
    onnx_model_path,
    audio_path,
    sample_rate=16000,
    n_mels=40,
    n_fft=400,
    hop_length=160,
    chunk_size_ms=500,
    use_running_stats=False,
    alpha=0.95
):
    """
    Test an ONNX model in streaming mode by processing chunks of audio
    Compatible with both GRU and LSTM models
    """
    print(f"Loading ONNX model from {onnx_model_path}")
    EP_list = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=EP_list)

    input_names = [input_meta.name for input_meta in session.get_inputs()]
    output_names = [output_meta.name for output_meta in session.get_outputs()]
    print(f"ONNX Model Inputs: {input_names}")
    print(f"ONNX Model Outputs: {output_names}")

    # Determine if this is an LSTM model (has cell state) or GRU model
    has_cell_state = 'cell_state_input' in input_names and 'cell_state_output' in output_names
    model_type = "LSTM" if has_cell_state else "GRU"
    print(f"Detected model type: {model_type}")

    hidden_state_input_meta = [im for im in session.get_inputs() if im.name == 'hidden_state_input'][0]
    hidden_dim = hidden_state_input_meta.shape[2]
    print(f"Hidden state dimension: {hidden_dim}")

    print(f"Loading audio from {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    print(f"Audio length: {len(audio)/sr:.2f} seconds")

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
    print(f"Processing in chunks of {chunk_size_ms}ms ({chunk_size_samples} samples)")
    print(f"Normalization method: {'Running statistics' if use_running_stats else 'Per-chunk statistics'}")

    batch_size = 1
    hidden_state = np.zeros((1, batch_size, hidden_dim), dtype=np.float32)

    # Initialize cell state for LSTM models
    if has_cell_state:
        cell_state = np.zeros((1, batch_size, hidden_dim), dtype=np.float32)

    all_predictions_raw_vectors = [] # List of (num_classes,) prediction vectors
    frame_predictions_list = []      # List of (num_classes,) vectors, repeated for frames

    running_mean, running_std = None, None
    num_classes = 0 # Determined from first chunk's output

    for chunk_start in tqdm(range(0, len(audio), chunk_size_samples), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size_samples, len(audio))
        chunk = audio[chunk_start:chunk_end]
        if len(chunk) == 0: continue

        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0)
        with torch.no_grad():
            spec = mel_spec(chunk_tensor)
            spec_db = amplitude_to_db(spec)
        spec_np = spec_db.numpy()

        current_mean, current_std = np.mean(spec_np), np.std(spec_np) + 1e-5
        if use_running_stats:
            if running_mean is None: running_mean, running_std = current_mean, current_std
            else:
                running_mean = alpha * running_mean + (1 - alpha) * current_mean
                running_std = alpha * running_std + (1 - alpha) * current_std
            spec_norm = (spec_np - running_mean) / running_std
        else:
            spec_norm = (spec_np - current_mean) / current_std

        spec_input = spec_norm.squeeze(0).astype(np.float32)
        spec_input = spec_input.reshape(1, n_mels, -1).astype(np.float32)

        # Prepare inputs and outputs based on model type
        if has_cell_state:
            # LSTM model
            ort_inputs = {
                'spectrogram_input': spec_input,
                'hidden_state_input': hidden_state,
                'cell_state_input': cell_state
            }
            output_names_to_run = ['output_logits', 'hidden_state_output', 'cell_state_output']
        else:
            # GRU model
            ort_inputs = {
                'spectrogram_input': spec_input,
                'hidden_state_input': hidden_state
            }
            output_names_to_run = ['output_logits', 'hidden_state_output']

        outputs = session.run(output_names_to_run, ort_inputs)
        prediction_logits_raw = outputs[0]
        hidden_state = outputs[1]

        # Update cell state for LSTM models
        if has_cell_state:
            cell_state = outputs[2]

        # Ensure prediction_logits_raw is numpy array for consistent processing
        if not isinstance(prediction_logits_raw, np.ndarray):
            prediction_logits_raw = np.array(prediction_logits_raw, dtype=np.float32)

        if num_classes == 0: # Determine num_classes from the first valid output
            if prediction_logits_raw.ndim == 0: # Scalar output
                num_classes = 1
            elif prediction_logits_raw.ndim > 0 :
                num_classes = prediction_logits_raw.shape[-1]
            else: # Should not happen
                print("Error: Could not determine num_classes from model output.")
                return {"predicted_class": -1, "prediction_probabilities": np.array([]), "method": "Error"}
            if num_classes == 0 : # If shape[-1] was 0 for some reason
                print("Warning: num_classes detected as 0 from model output shape, defaulting to 1.")
                num_classes = 1


        # Extract a (num_classes,) prediction vector for the current chunk
        current_chunk_pred_vector = np.zeros(num_classes, dtype=np.float32) # Default
        if prediction_logits_raw.ndim == 0: # Scalar output
             current_chunk_pred_vector = np.array([prediction_logits_raw.item()], dtype=np.float32)
        elif prediction_logits_raw.ndim == 1: # (C,)
            current_chunk_pred_vector = prediction_logits_raw
        elif prediction_logits_raw.ndim == 2: # (batch, C), e.g. (1,C)
            current_chunk_pred_vector = prediction_logits_raw[0, :]
        elif prediction_logits_raw.ndim == 3: # (batch, seq, C), e.g. (1,1,C)
            current_chunk_pred_vector = prediction_logits_raw[0, 0, :]
        else:
            print(f"Warning: Unexpected prediction_logits_raw dimension: {prediction_logits_raw.shape}. Using zeros.")

        # Ensure vector has correct length if num_classes was adjusted or misaligned
        if len(current_chunk_pred_vector) != num_classes and num_classes==1: # common for scalar becoming array
             current_chunk_pred_vector = np.array([current_chunk_pred_vector.sum() / len(current_chunk_pred_vector if len(current_chunk_pred_vector)>0 else [1]) ], dtype=np.float32) # Average if mismatch, then make (1,)
        elif len(current_chunk_pred_vector) != num_classes :
            print(f"Warning: Mismatch between extracted vector length {len(current_chunk_pred_vector)} and num_classes {num_classes}. Adjusting.")
            # Simple fix: take first num_classes or pad with mean
            val_to_pad = np.mean(current_chunk_pred_vector) if len(current_chunk_pred_vector) > 0 else (1.0/num_classes if num_classes > 0 else 0)
            temp_vec = np.full(num_classes, val_to_pad, dtype=np.float32)
            copy_len = min(len(current_chunk_pred_vector), num_classes)
            if copy_len > 0:
                temp_vec[:copy_len] = current_chunk_pred_vector[:copy_len]
            current_chunk_pred_vector = temp_vec


        all_predictions_raw_vectors.append(current_chunk_pred_vector)

        num_frames_in_audio_chunk = chunk.shape[0] // hop_length
        for _ in range(num_frames_in_audio_chunk):
            frame_predictions_list.append(current_chunk_pred_vector)

    # After loop
    if not all_predictions_raw_vectors:
        _nc_fallback = num_classes if num_classes > 0 else 1
        final_prediction_probabilities = np.full(_nc_fallback, 1.0/_nc_fallback, dtype=np.float32)
        if num_classes == 0: final_prediction_probabilities = np.array([], dtype=np.float32) # Truly empty if no info
        predicted_class_overall = np.argmax(final_prediction_probabilities) if final_prediction_probabilities.size > 0 else -1
        print("Warning: No predictions were made (e.g., audio too short or no chunks processed).")
    else:
        stacked_chunk_preds = np.array(all_predictions_raw_vectors) # Shape (num_chunks, num_classes)
        # Handle case where num_classes is 1 and stacked_chunk_preds might be (N,) instead of (N,1)
        if stacked_chunk_preds.ndim == 1 and num_classes == 1:
            stacked_chunk_preds = stacked_chunk_preds.reshape(-1, 1)

        if stacked_chunk_preds.shape[1] != num_classes : # Should not happen if logic above is correct
            print(f"Error: Dimension mismatch in stacked predictions. Expected {num_classes} classes, got {stacked_chunk_preds.shape[1]}")
            # Fallback for final probabilities to avoid crash
            _nc_fallback = num_classes if num_classes > 0 else stacked_chunk_preds.shape[1] if stacked_chunk_preds.ndim > 1 and stacked_chunk_preds.shape[1] > 0 else 1
            final_prediction_probabilities = np.full(_nc_fallback, 1.0/_nc_fallback, dtype=np.float32)
        else:
             final_prediction_probabilities = np.mean(stacked_chunk_preds, axis=0) # Shape (num_classes,)
        predicted_class_overall = np.argmax(final_prediction_probabilities) if final_prediction_probabilities.size > 0 else -1

    print(f"Final prediction across all chunks: Class {predicted_class_overall}")
    print(f"Prediction probabilities: {final_prediction_probabilities}") # Should be (num_classes,)

    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(audio))/sample_rate, audio)
    plt.title("Audio Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")

    max_plot_frames = len(audio) // hop_length
    frame_predictions_np = np.array([])
    if frame_predictions_list: # frame_predictions_list contains (C,) arrays
        frame_predictions_np = np.array(frame_predictions_list[:max_plot_frames]) # Results in (total_plot_frames, num_classes)

    plt.subplot(3, 1, 2)
    if frame_predictions_np.size > 0 and num_classes > 0:
        time_axis_preds = np.arange(frame_predictions_np.shape[0]) * hop_length / sample_rate
        for i in range(num_classes):
            plt.plot(time_axis_preds, frame_predictions_np[:, i], label=f"Class {i}")
        plt.title("Predictions Over Time (Probabilities per Class)"); plt.xlabel("Time (s)")
        plt.ylabel("Probability"); plt.legend()
    else:
        plt.text(0.5,0.5, "No prediction data for class probabilities.", transform=plt.gca().transAxes, ha='center',va='center')

    plt.subplot(3, 1, 3)
    if frame_predictions_np.size > 0 and num_classes > 0:
        max_confidences = np.max(frame_predictions_np, axis=1)
        predicted_classes_at_frame = np.argmax(frame_predictions_np, axis=1)
        time_axis_confidence = np.arange(len(max_confidences)) * hop_length / sample_rate

        cmap_scatter = plt.cm.get_cmap('tab10', num_classes if num_classes > 0 else 1)
        plt.scatter(time_axis_confidence, max_confidences, c=predicted_classes_at_frame, cmap=cmap_scatter,
                    vmin=-0.5, vmax=(num_classes - 0.5 if num_classes > 0 else 0.5), alpha=0.7, s=10)

        legend_elements = []
        unique_scatter_classes = np.unique(predicted_classes_at_frame)
        for i in range(num_classes):
            if i in unique_scatter_classes:
                legend_elements.append(plt.Line2D([0],[0], marker='o', color='w', label=f'Class {i}',
                                       markerfacecolor=cmap_scatter(i / (num_classes if num_classes > 1 else 1.0) if num_classes > 0 else 0.0), markersize=8))

        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Confidence Threshold (0.5)')
        if legend_elements or plt.gca().get_legend_handles_labels()[0]: # Add threshold to legend if other elements exist
            handles, labels = plt.gca().get_legend_handles_labels()
            if not any("Confidence Threshold" in lab for lab in labels): # Add only if not already there from scatter legend
                 handles.extend([h for h in plt.gca().get_lines() if h.get_label() == 'Confidence Threshold (0.5)'])
            plt.legend(handles=handles + legend_elements if not legend_elements else legend_elements)


        plt.title("Prediction Confidence Over Time (Highest Class)"); plt.xlabel("Time (s)")
        plt.ylabel("Max Confidence"); plt.ylim(0, 1.05)
    else:
        plt.text(0.5,0.5, "No prediction data for confidence plot.", transform=plt.gca().transAxes, ha='center',va='center')

    plt.tight_layout()
    stats_type = "running_stats" if use_running_stats else "per_chunk_stats"
    alpha_str = f"{alpha:.2f}" if use_running_stats else "NA"
    model_type_str = model_type.lower()
    output_filename = f"streaming_predictions_{model_type_str}_{stats_type}_{chunk_size_ms}ms_alpha{alpha_str}.png"
    plt.savefig(output_filename); plt.show()
    print(f"Test complete! Check {output_filename} for visualization.")

    return {"predicted_class": predicted_class_overall,
            "prediction_probabilities": final_prediction_probabilities, # This is now (num_classes,)
            "method": "Running Stats" if use_running_stats else "Per-chunk Stats",
            "model_type": model_type}

def compare_normalization_methods(onnx_model_path, audio_path, chunk_size_ms=500, **kwargs):
    print("\n===== Testing with per-chunk statistics =====")
    per_chunk_results = test_onnx_streaming(onnx_model_path, audio_path, chunk_size_ms=chunk_size_ms, use_running_stats=False, **kwargs)

    print("\n===== Testing with running statistics =====")
    running_stats_results = test_onnx_streaming(onnx_model_path, audio_path, chunk_size_ms=chunk_size_ms, use_running_stats=True, **kwargs)

    print("\n===== Comparison Results =====")
    print(f"Model type: {per_chunk_results.get('model_type', 'Unknown')}")
    print(f"Chunk size: {chunk_size_ms}ms")
    alpha_val = kwargs.get('alpha', 'N/A')
    print(f"Alpha for running stats: {alpha_val}")
    print(f"Per-chunk stats prediction: Class {per_chunk_results['predicted_class']}")
    print(f"Running stats prediction: Class {running_stats_results['predicted_class']}")

    # Ensure probabilities are iterable and have content
    pc_probs = per_chunk_results.get('prediction_probabilities', np.array([]))
    rs_probs = running_stats_results.get('prediction_probabilities', np.array([]))

    if pc_probs.size == 0 and rs_probs.size == 0:
        print("\nNo probability data to compare.")
        return {"per_chunk": per_chunk_results, "running_stats": running_stats_results}

    # Determine max length for zipping if arrays have different lengths (e.g. one errored out)
    max_len = 0
    if hasattr(pc_probs, '__len__'): max_len = max(max_len, len(pc_probs))
    if hasattr(rs_probs, '__len__'): max_len = max(max_len, len(rs_probs))

    if max_len == 0 and (isinstance(pc_probs, np.ndarray) and pc_probs.ndim == 0) and \
                       (isinstance(rs_probs, np.ndarray) and rs_probs.ndim == 0) : # Both are scalars
        max_len = 1 # Allow zipping scalar as single element list

    print("\nProbability comparison (final averaged):")
    for i in range(max_len):
        val_pc = pc_probs[i] if hasattr(pc_probs, '__getitem__') and i < len(pc_probs) else (pc_probs.item() if max_len==1 and hasattr(pc_probs, 'item') else np.nan)
        val_rs = rs_probs[i] if hasattr(rs_probs, '__getitem__') and i < len(rs_probs) else (rs_probs.item() if max_len==1 and hasattr(rs_probs, 'item') else np.nan)
        diff = val_rs - val_pc
        print(f"Class {i}: Per-chunk: {val_pc:.4f}, Running: {val_rs:.4f}, Diff: {diff:+.4f}")

    return {"per_chunk": per_chunk_results, "running_stats": running_stats_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX keyword spotting model in streaming mode")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size in milliseconds")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--n_mels", type=int, default=40, help="Number of mel bands")
    parser.add_argument("--n_fft", type=int, default=400, help="FFT size for Mel Spectrogram")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length for Mel Spectrogram")
    parser.add_argument("--compare", action="store_true", help="Compare both normalization methods")
    parser.add_argument("--running_stats", action="store_true", help="Use running statistics for normalization (if not comparing)")
    parser.add_argument("--alpha", type=float, default=0.5, help="EMA coefficient for running statistics (0.0 to 1.0)")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    kwargs_for_streaming = {
        "sample_rate": args.sample_rate, "n_mels": args.n_mels,
        "n_fft": args.n_fft, "hop_length": args.hop_length, "alpha": args.alpha
    }

    if args.compare:
        compare_normalization_methods(args.model, args.audio, chunk_size_ms=args.chunk_size, **kwargs_for_streaming)
    else:
        test_onnx_streaming(args.model, args.audio, chunk_size_ms=args.chunk_size, use_running_stats=args.running_stats, **kwargs_for_streaming)
