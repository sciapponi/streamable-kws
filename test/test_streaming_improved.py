import numpy as np
import onnxruntime as ort
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchaudio
import argparse


def test_onnx_streaming_improved(
    onnx_model_path,
    audio_path,
    sample_rate=16000,
    n_mels=40,
    n_fft=400,
    hop_length=160,
    chunk_size_ms=200,  # Fixed window size
    step_size_ms=100,   # Increased step size for less overlap
    use_running_stats=True,
    alpha=0.95,
    reset_hidden_state_every_n_chunks=10,  # Reset state periodically
    temporal_smoothing=True,
    smoothing_window=5
):
    """
    Improved version of streaming inference with configurable step size
    and better state management
    """
    print(f"Loading ONNX model from {onnx_model_path}")
    EP_list = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=EP_list)

    input_names = [input_meta.name for input_meta in session.get_inputs()]
    output_names = [output_meta.name for output_meta in session.get_outputs()]
    print(f"ONNX Model Inputs: {input_names}")
    print(f"ONNX Model Outputs: {output_names}")

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
    step_size_samples = int(sample_rate * step_size_ms / 1000)
    print(f"Processing in sliding windows: chunk size {chunk_size_ms}ms every {step_size_ms}ms")
    print(f"Normalization method: {'Running statistics' if use_running_stats else 'Per-chunk statistics'}")
    print(f"Hidden state reset: Every {reset_hidden_state_every_n_chunks} chunks")
    print(f"Temporal smoothing: {'Enabled' if temporal_smoothing else 'Disabled'}")

    batch_size = 1
    hidden_state = np.zeros((1, batch_size, hidden_dim), dtype=np.float32)

    all_predictions_raw_vectors = []  # List of (num_classes,) prediction vectors
    frame_predictions_list = []       # List of (num_classes,) vectors, repeated for frames
    prediction_timestamps = []        # List of timestamps for each prediction (center of window)

    running_mean, running_std = None, None
    num_classes = 0  # Determined from first chunk's output
    chunk_counter = 0

    for chunk_start in tqdm(range(0, len(audio) - chunk_size_samples + 1, step_size_samples), desc="Processing chunks"):
        chunk_end = chunk_start + chunk_size_samples
        chunk = audio[chunk_start:chunk_end]
        if len(chunk) == 0: continue

        # Reset hidden state periodically
        chunk_counter += 1
        if reset_hidden_state_every_n_chunks > 0 and chunk_counter % reset_hidden_state_every_n_chunks == 0:
            hidden_state = np.zeros((1, batch_size, hidden_dim), dtype=np.float32)
            print(f"Hidden state reset at chunk {chunk_counter}")

        # Calculate timestamp for this window (center point)
        window_center_time = (chunk_start + chunk_size_samples/2) / sample_rate
        prediction_timestamps.append(window_center_time)

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

        outputs = session.run(['output_logits', 'hidden_state_output'],
                             {'spectrogram_input': spec_input, 'hidden_state_input': hidden_state})
        prediction_logits_raw = outputs[0]
        hidden_state = outputs[1]

        # Ensure prediction_logits_raw is numpy array for consistent processing
        if not isinstance(prediction_logits_raw, np.ndarray):
            prediction_logits_raw = np.array(prediction_logits_raw, dtype=np.float32)

        if num_classes == 0:  # Determine num_classes from the first valid output
            if prediction_logits_raw.ndim == 0:  # Scalar output
                num_classes = 1
            elif prediction_logits_raw.ndim > 0:
                num_classes = prediction_logits_raw.shape[-1]
            else:  # Should not happen
                print("Error: Could not determine num_classes from model output.")
                return {"predicted_class": -1, "prediction_probabilities": np.array([]), "method": "Error"}
            if num_classes == 0:  # If shape[-1] was 0 for some reason
                print("Warning: num_classes detected as 0 from model output shape, defaulting to 1.")
                num_classes = 1

        # Extract a (num_classes,) prediction vector for the current chunk
        current_chunk_pred_vector = np.zeros(num_classes, dtype=np.float32)  # Default
        if prediction_logits_raw.ndim == 0:  # Scalar output
             current_chunk_pred_vector = np.array([prediction_logits_raw.item()], dtype=np.float32)
        elif prediction_logits_raw.ndim == 1:  # (C,)
            current_chunk_pred_vector = prediction_logits_raw
        elif prediction_logits_raw.ndim == 2:  # (batch, C), e.g. (1,C)
            current_chunk_pred_vector = prediction_logits_raw[0, :]
        elif prediction_logits_raw.ndim == 3:  # (batch, seq, C), e.g. (1,1,C)
            current_chunk_pred_vector = prediction_logits_raw[0, 0, :]
        else:
            print(f"Warning: Unexpected prediction_logits_raw dimension: {prediction_logits_raw.shape}. Using zeros.")

        # Ensure vector has correct length if num_classes was adjusted or misaligned
        if len(current_chunk_pred_vector) != num_classes and num_classes == 1:
             current_chunk_pred_vector = np.array([current_chunk_pred_vector.sum() / len(current_chunk_pred_vector if len(current_chunk_pred_vector) > 0 else [1])], dtype=np.float32)
        elif len(current_chunk_pred_vector) != num_classes:
            print(f"Warning: Mismatch between vector length {len(current_chunk_pred_vector)} and num_classes {num_classes}. Adjusting.")
            val_to_pad = np.mean(current_chunk_pred_vector) if len(current_chunk_pred_vector) > 0 else (1.0/num_classes if num_classes > 0 else 0)
            temp_vec = np.full(num_classes, val_to_pad, dtype=np.float32)
            copy_len = min(len(current_chunk_pred_vector), num_classes)
            if copy_len > 0:
                temp_vec[:copy_len] = current_chunk_pred_vector[:copy_len]
            current_chunk_pred_vector = temp_vec

        all_predictions_raw_vectors.append(current_chunk_pred_vector)

        # For plotting, we'll associate this prediction with frames
        num_frames_in_step = step_size_samples // hop_length
        for _ in range(num_frames_in_step):
            frame_predictions_list.append(current_chunk_pred_vector)

    # After loop - Apply temporal smoothing if enabled
    if temporal_smoothing and len(all_predictions_raw_vectors) > 0:
        smoothed_predictions = []
        pred_array = np.array(all_predictions_raw_vectors)

        for i in range(len(pred_array)):
            # Define window boundaries
            window_start = max(0, i - smoothing_window // 2)
            window_end = min(len(pred_array), i + smoothing_window // 2 + 1)

            # Calculate weighted average (center has higher weight)
            weights = np.ones(window_end - window_start)
            weights = weights / weights.sum()  # Normalize

            # Apply weighted average
            smoothed_pred = np.average(pred_array[window_start:window_end], axis=0, weights=weights)
            smoothed_predictions.append(smoothed_pred)

        all_predictions_raw_vectors = smoothed_predictions

    # Calculate final prediction
    if not all_predictions_raw_vectors:
        _nc_fallback = num_classes if num_classes > 0 else 1
        final_prediction_probabilities = np.full(_nc_fallback, 1.0/_nc_fallback, dtype=np.float32)
        if num_classes == 0: final_prediction_probabilities = np.array([], dtype=np.float32)
        predicted_class_overall = np.argmax(final_prediction_probabilities) if final_prediction_probabilities.size > 0 else -1
        print("Warning: No predictions were made (e.g., audio too short or no chunks processed).")
    else:
        stacked_chunk_preds = np.array(all_predictions_raw_vectors)
        # Handle case where num_classes is 1 and stacked_chunk_preds might be (N,) instead of (N,1)
        if stacked_chunk_preds.ndim == 1 and num_classes == 1:
            stacked_chunk_preds = stacked_chunk_preds.reshape(-1, 1)

        if stacked_chunk_preds.shape[1] != num_classes:
            print(f"Error: Dimension mismatch in stacked predictions. Expected {num_classes} classes, got {stacked_chunk_preds.shape[1]}")
            _nc_fallback = num_classes if num_classes > 0 else stacked_chunk_preds.shape[1] if stacked_chunk_preds.ndim > 1 and stacked_chunk_preds.shape[1] > 0 else 1
            final_prediction_probabilities = np.full(_nc_fallback, 1.0/_nc_fallback, dtype=np.float32)
        else:
             final_prediction_probabilities = np.mean(stacked_chunk_preds, axis=0)
        predicted_class_overall = np.argmax(final_prediction_probabilities) if final_prediction_probabilities.size > 0 else -1

    print(f"Final prediction across all chunks: Class {predicted_class_overall}")
    print(f"Prediction probabilities: {final_prediction_probabilities}")

    # Create comprehensive visualization
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(audio))/sample_rate, audio)
    plt.title("Audio Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")

    # Plot predictions at their actual timestamps (window centers)
    plt.subplot(3, 1, 2)
    if len(all_predictions_raw_vectors) > 0 and num_classes > 0:
        predictions_np = np.array(all_predictions_raw_vectors)
        for i in range(num_classes):
            plt.plot(prediction_timestamps[:len(predictions_np)], predictions_np[:, i],
                    label=f"Class {i}", marker='o', markersize=3, linestyle='-', alpha=0.7)
        plt.title(f"Predictions Over Time (Window Centers) - {'Smoothed' if temporal_smoothing else 'Raw'}")
        plt.xlabel("Time (s)"); plt.ylabel("Probability"); plt.legend()
    else:
        plt.text(0.5, 0.5, "No prediction data for class probabilities.",
                transform=plt.gca().transAxes, ha='center', va='center')

    # Plot max confidence at each window
    plt.subplot(3, 1, 3)
    if len(all_predictions_raw_vectors) > 0 and num_classes > 0:
        max_confidences = np.max(predictions_np, axis=1)
        predicted_classes_at_window = np.argmax(predictions_np, axis=1)

        cmap_scatter = plt.cm.get_cmap('tab10', num_classes if num_classes > 0 else 1)
        plt.scatter(prediction_timestamps[:len(max_confidences)], max_confidences,
                   c=predicted_classes_at_window, cmap=cmap_scatter,
                   vmin=-0.5, vmax=(num_classes - 0.5 if num_classes > 0 else 0.5),
                   alpha=0.7, s=20)

        legend_elements = []
        unique_scatter_classes = np.unique(predicted_classes_at_window)
        for i in range(num_classes):
            if i in unique_scatter_classes:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Class {i}',
                                       markerfacecolor=cmap_scatter(i / (num_classes if num_classes > 1 else 1.0) if num_classes > 0 else 0.0), markersize=8))

        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Confidence Threshold (0.5)')
        if legend_elements or plt.gca().get_legend_handles_labels()[0]:
            handles, labels = plt.gca().get_legend_handles_labels()
            if not any("Confidence Threshold" in lab for lab in labels):
                 handles.extend([h for h in plt.gca().get_lines() if h.get_label() == 'Confidence Threshold (0.5)'])
            plt.legend(handles=handles + legend_elements if not legend_elements else legend_elements)

        plt.title("Prediction Confidence at Each Window Center"); plt.xlabel("Time (s)")
        plt.ylabel("Max Confidence"); plt.ylim(0, 1.05)
    else:
        plt.text(0.5, 0.5, "No prediction data for confidence plot.",
                transform=plt.gca().transAxes, ha='center', va='center')

    plt.tight_layout()
    stats_type = "running_stats" if use_running_stats else "per_chunk_stats"
    alpha_str = f"{alpha:.2f}" if use_running_stats else "NA"
    smooth_str = "smoothed" if temporal_smoothing else "raw"
    reset_str = f"reset{reset_hidden_state_every_n_chunks}"
    output_filename = f"improved_sliding_window_{stats_type}_{chunk_size_ms}ms_step{step_size_ms}ms_alpha{alpha_str}_{smooth_str}_{reset_str}.png"
    plt.savefig(output_filename); plt.show()
    print(f"Test complete! Check {output_filename} for visualization.")

    return {"predicted_class": predicted_class_overall,
            "prediction_probabilities": final_prediction_probabilities,
            "method": f"{'Running' if use_running_stats else 'Per-chunk'} Stats, {'Smoothed' if temporal_smoothing else 'Raw'}, Reset={reset_hidden_state_every_n_chunks}"}


def compare_methods(onnx_model_path, audio_path, chunk_size_ms=200, **kwargs):
    """Compare different streaming configurations"""
    results = {}

    # Basic configurations to test
    configs = [
        {"name": "Baseline (20ms step)", "step_size_ms": 20, "use_running_stats": True,
         "reset_hidden_state_every_n_chunks": 0, "temporal_smoothing": False},

        {"name": "Larger step (100ms)", "step_size_ms": 100, "use_running_stats": True,
         "reset_hidden_state_every_n_chunks": 0, "temporal_smoothing": False},

        {"name": "With state reset", "step_size_ms": 100, "use_running_stats": True,
         "reset_hidden_state_every_n_chunks": 5, "temporal_smoothing": False},

        {"name": "With smoothing", "step_size_ms": 100, "use_running_stats": True,
         "reset_hidden_state_every_n_chunks": 5, "temporal_smoothing": True}
    ]

    for config in configs:
        print(f"\n===== Testing {config['name']} =====")
        results[config['name']] = test_onnx_streaming_improved(
            onnx_model_path,
            audio_path,
            chunk_size_ms=chunk_size_ms,
            step_size_ms=config['step_size_ms'],
            use_running_stats=config['use_running_stats'],
            reset_hidden_state_every_n_chunks=config['reset_hidden_state_every_n_chunks'],
            temporal_smoothing=config['temporal_smoothing'],
            **kwargs
        )

    # Print comparison
    print("\n===== Results Summary =====")
    for name, result in results.items():
        print(f"{name}: Class {result['predicted_class']}")

    return results


def discrete_chunk_classification(onnx_model_path, audio_path, chunk_size_ms=200, **kwargs):
    """Non-overlapping chunk classification for comparison with streaming"""
    print("\n===== Running discrete chunk classification (no overlap) =====")

    # Load model
    EP_list = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=EP_list)

    # Load audio
    sample_rate = kwargs.get('sample_rate', 16000)
    audio, sr = librosa.load(audio_path, sr=sample_rate)

    # Feature extraction parameters
    n_mels = kwargs.get('n_mels', 40)
    n_fft = kwargs.get('n_fft', 400)
    hop_length = kwargs.get('hop_length', 160)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    # Calculate chunk size in samples
    chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)

    # Process each discrete chunk
    all_predictions = []
    chunk_timestamps = []

    for chunk_start in range(0, len(audio), chunk_size_samples):
        chunk_end = min(chunk_start + chunk_size_samples, len(audio))
        if chunk_end - chunk_start < chunk_size_samples / 2:  # Skip very short chunks
            continue

        chunk = audio[chunk_start:chunk_end]
        chunk_time = (chunk_start + (chunk_end - chunk_start)/2) / sample_rate
        chunk_timestamps.append(chunk_time)

        # Get features
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0)
        with torch.no_grad():
            spec = mel_spec(chunk_tensor)
            spec_db = amplitude_to_db(spec)
        spec_np = spec_db.numpy()

        # Normalize
        mean, std = np.mean(spec_np), np.std(spec_np) + 1e-5
        spec_norm = (spec_np - mean) / std

        spec_input = spec_norm.squeeze(0).astype(np.float32)
        spec_input = spec_input.reshape(1, n_mels, -1).astype(np.float32)

        # Initialize hidden state (new for each chunk)
        hidden_dim = [im for im in session.get_inputs() if im.name == 'hidden_state_input'][0].shape[2]
        hidden_state = np.zeros((1, 1, hidden_dim), dtype=np.float32)

        # Get prediction
        outputs = session.run(['output_logits', 'hidden_state_output'],
                           {'spectrogram_input': spec_input, 'hidden_state_input': hidden_state})
        pred = outputs[0]

        # Process prediction to consistent format
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)

        if pred.ndim == 3:  # (batch, seq, classes)
            pred = pred[0, 0]
        elif pred.ndim == 2:  # (batch, classes)
            pred = pred[0]

        all_predictions.append(pred)

    # Calculate overall prediction
    if not all_predictions:
        print("No chunks processed!")
        return {"predicted_class": -1, "prediction_probabilities": np.array([])}

    predictions_array = np.array(all_predictions)
    avg_predictions = np.mean(predictions_array, axis=0)
    predicted_class = np.argmax(avg_predictions)

    print(f"Discrete chunks prediction: Class {predicted_class}")
    print(f"Prediction probabilities: {avg_predictions}")

    # Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(audio))/sample_rate, audio)
    plt.title("Audio Waveform"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    for i in range(predictions_array.shape[1]):
        plt.plot(chunk_timestamps, predictions_array[:, i],
                label=f"Class {i}", marker='o', linestyle='-')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.title("Discrete Chunk Predictions"); plt.xlabel("Time (s)")
    plt.ylabel("Probability"); plt.legend()

    plt.tight_layout()
    plt.savefig("discrete_chunk_predictions.png")
    plt.show()

    return {"predicted_class": predicted_class,
            "prediction_probabilities": avg_predictions,
            "method": "Discrete Chunks"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX keyword spotting model in streaming mode")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size in milliseconds")
    parser.add_argument("--step_size", type=int, default=100, help="Step size in milliseconds")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--n_mels", type=int, default=40, help="Number of mel bands")
    parser.add_argument("--n_fft", type=int, default=400, help="FFT size for Mel Spectrogram")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length for Mel Spectrogram")
    parser.add_argument("--compare", action="store_true", help="Compare different methods")
    parser.add_argument("--discrete", action="store_true", help="Run discrete chunk classification")
    parser.add_argument("--reset_state", type=int, default=0, help="Reset hidden state every N chunks (0 for no reset)")
    parser.add_argument("--temporal_smoothing", action="store_true", help="Apply temporal smoothing")
    parser.add_argument("--alpha", type=float, default=0.95, help="EMA coefficient for running statistics (0.0 to 1.0)")
    args = parser.parse_args()

    kwargs_common = {
        "sample_rate": args.sample_rate,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "alpha": args.alpha
    }

    if args.compare:
        compare_methods(args.model, args.audio, chunk_size_ms=args.chunk_size, **kwargs_common)
    elif args.discrete:
        discrete_chunk_classification(args.model, args.audio, chunk_size_ms=args.chunk_size, **kwargs_common)
    else:
        test_onnx_streaming_improved(
            args.model,
            args.audio,
            chunk_size_ms=args.chunk_size,
            step_size_ms=args.step_size,
            reset_hidden_state_every_n_chunks=args.reset_state,
            temporal_smoothing=args.temporal_smoothing,
            **kwargs_common
        )
