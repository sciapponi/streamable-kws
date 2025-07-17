import numpy as np
import onnxruntime as ort
import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio
import argparse
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm


def load_audio(file_path, sample_rate=16000):
    """Load audio file at the given sample rate"""
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr


class StreamingEvaluator:
    def __init__(
        self,
        onnx_model_path,
        test_dir,
        class_names,
        sample_rate=16000,
        n_mels=40,
        n_fft=400,
        hop_length=160,
        chunk_size_ms=500,
        use_running_stats=False,
        alpha=0.95,
        output_dir="results"
    ):
        self.onnx_model_path = onnx_model_path
        self.test_dir = test_dir
        self.class_names = class_names
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_size_ms = chunk_size_ms
        self.use_running_stats = use_running_stats
        self.alpha = alpha
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Setup ONNX session
        self.setup_onnx_session()

        # Setup spectrogram transforms
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Chunk size in samples
        self.chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)

        # Results storage
        self.clip_true_labels = []
        self.clip_predicted_labels = []
        self.clip_confidences = []

        self.chunk_true_labels = []
        self.chunk_predicted_labels = []
        self.chunk_confidences = []

    def setup_onnx_session(self):
        """Initialize the ONNX runtime session"""
        print(f"Loading ONNX model from {self.onnx_model_path}")
        EP_list = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_model_path, providers=EP_list)

        self.input_names = [input_meta.name for input_meta in self.session.get_inputs()]
        self.output_names = [output_meta.name for output_meta in self.session.get_outputs()]
        print(f"ONNX Model Inputs: {self.input_names}")
        print(f"ONNX Model Outputs: {self.output_names}")

        # Check if model has hidden state input (for stateful models)
        self.has_hidden_state = any('hidden_state' in input_name for input_name in self.input_names)

        if self.has_hidden_state:
            hidden_state_input_meta = [im for im in self.session.get_inputs() if 'hidden_state' in im.name][0]
            self.hidden_dim = hidden_state_input_meta.shape[2]
            print(f"Model has hidden state with dimension: {self.hidden_dim}")
        else:
            self.hidden_dim = 0
            print("Model does not have hidden state input")

    def get_audio_files_and_labels(self):
        """Get all audio files and their corresponding labels from the test directory"""
        audio_files = []
        labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.test_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist!")
                continue

            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                    if f.endswith(('.wav', '.mp3', '.flac'))]
            audio_files.extend(files)
            labels.extend([class_idx] * len(files))

        return audio_files, labels

    def process_audio_chunk(self, chunk, running_mean=None, running_std=None):
        """Process a single audio chunk and return normalized spectrogram"""
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0)
        with torch.no_grad():
            spec = self.mel_spec(chunk_tensor)
            spec_db = self.amplitude_to_db(spec)
        spec_np = spec_db.numpy()

        current_mean, current_std = np.mean(spec_np), np.std(spec_np) + 1e-5

        if self.use_running_stats and running_mean is not None:
            running_mean = self.alpha * running_mean + (1 - self.alpha) * current_mean
            running_std = self.alpha * running_std + (1 - self.alpha) * current_std
            spec_norm = (spec_np - running_mean) / running_std
        else:
            # Use per-chunk stats or initialize running stats
            spec_norm = (spec_np - current_mean) / current_std
            if self.use_running_stats:
                running_mean, running_std = current_mean, current_std

        spec_input = spec_norm.squeeze(0).astype(np.float32)
        spec_input = spec_input.reshape(1, self.n_mels, -1).astype(np.float32)

        return spec_input, running_mean, running_std

    def process_audio_file(self, audio_file, true_label):
        """Process an audio file in chunks, accumulating predictions"""
        audio, _ = load_audio(audio_file, self.sample_rate)

        # Initial hidden state (if model uses it)
        hidden_state = None
        if self.has_hidden_state:
            hidden_state = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)

        # For accumulating chunk predictions
        chunk_predictions = []
        chunk_confidences = []

        # For running statistics normalization
        running_mean, running_std = None, None

        for chunk_start in range(0, len(audio), self.chunk_size_samples):
            chunk_end = min(chunk_start + self.chunk_size_samples, len(audio))
            chunk = audio[chunk_start:chunk_end]

            if len(chunk) < self.hop_length:  # Skip chunks that are too small
                continue

            # Process chunk to get normalized spectrogram
            spec_input, running_mean, running_std = self.process_audio_chunk(
                chunk, running_mean, running_std
            )

            # Prepare inputs based on model requirements
            inputs = {'spectrogram_input': spec_input}
            output_names_to_fetch = ['output_logits']

            if self.has_hidden_state:
                inputs['hidden_state_input'] = hidden_state
                output_names_to_fetch.append('hidden_state_output')

            # Run inference
            outputs = self.session.run(output_names_to_fetch, inputs)
            logits = outputs[0]

            # Update hidden state if model uses it
            if self.has_hidden_state:
                hidden_state = outputs[1]

            # Extract predictions
            if isinstance(logits, np.ndarray):
                if logits.ndim == 3:  # (batch, seq, classes)
                    logits = logits[0, 0]
                elif logits.ndim == 2:  # (batch, classes)
                    logits = logits[0]

            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()

            pred_class = np.argmax(probabilities)
            confidence = probabilities[pred_class]

            # Store chunk-level results
            chunk_predictions.append(pred_class)
            chunk_confidences.append(confidence)
            self.chunk_true_labels.append(true_label)
            self.chunk_predicted_labels.append(pred_class)
            self.chunk_confidences.append(confidence)

        # Determine clip-level prediction by voting
        if chunk_predictions:
            # Vote by class with highest average confidence
            class_avg_conf = {}
            for cls, conf in zip(chunk_predictions, chunk_confidences):
                if cls not in class_avg_conf:
                    class_avg_conf[cls] = []
                class_avg_conf[cls].append(conf)

            for cls in class_avg_conf:
                class_avg_conf[cls] = np.mean(class_avg_conf[cls])

            # Get class with highest average confidence
            clip_prediction = max(class_avg_conf.items(), key=lambda x: x[1])[0]
            clip_confidence = class_avg_conf[clip_prediction]
        else:
            # Fallback if no chunks were processed
            clip_prediction = 0  # Default to first class
            clip_confidence = 0.0

        return clip_prediction, clip_confidence

    def evaluate_test_set(self):
        """Evaluate the model on the entire test set"""
        audio_files, true_labels = self.get_audio_files_and_labels()

        print(f"Found {len(audio_files)} test files across {len(self.class_names)} classes")
        print(f"Processing in chunks of {self.chunk_size_ms}ms, using {'running' if self.use_running_stats else 'per-chunk'} statistics")

        # Process each audio file
        for i, (audio_file, true_label) in enumerate(tqdm(zip(audio_files, true_labels), total=len(audio_files), desc="Processing audio files")):
            clip_prediction, clip_confidence = self.process_audio_file(audio_file, true_label)

            # Store clip-level results
            self.clip_true_labels.append(true_label)
            self.clip_predicted_labels.append(clip_prediction)
            self.clip_confidences.append(clip_confidence)

    def compute_metrics(self):
        """Compute and report metrics for both clip and chunk level predictions"""
        # Clip-level metrics
        clip_accuracy = accuracy_score(self.clip_true_labels, self.clip_predicted_labels)
        clip_cm = confusion_matrix(self.clip_true_labels, self.clip_predicted_labels, labels=range(len(self.class_names)))

        print("\n===== Clip-Level Metrics =====")
        print(f"Accuracy: {clip_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.clip_true_labels, self.clip_predicted_labels, target_names=self.class_names))

        # Chunk-level metrics
        chunk_accuracy = accuracy_score(self.chunk_true_labels, self.chunk_predicted_labels)
        chunk_cm = confusion_matrix(self.chunk_true_labels, self.chunk_predicted_labels, labels=range(len(self.class_names)))

        print("\n===== Chunk-Level Metrics =====")
        print(f"Accuracy: {chunk_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.chunk_true_labels, self.chunk_predicted_labels, target_names=self.class_names))

        # Plot confusion matrices
        self._plot_confusion_matrix(clip_cm, "Clip-Level Confusion Matrix", "clip_confusion_matrix.png")
        self._plot_confusion_matrix(chunk_cm, "Chunk-Level Confusion Matrix", "chunk_confusion_matrix.png")

        # Plot confidence distributions by class
        self._plot_confidence_distributions()

        return {
            "clip_accuracy": clip_accuracy,
            "chunk_accuracy": chunk_accuracy,
            "clip_confusion_matrix": clip_cm,
            "chunk_confusion_matrix": chunk_cm
        }

    def _plot_confusion_matrix(self, cm, title, filename):
        """Plot and save a confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _plot_confidence_distributions(self):
        """Plot confidence distributions for correct and incorrect predictions"""
        # Clip-level confidence distributions
        clip_correct = np.array(self.clip_true_labels) == np.array(self.clip_predicted_labels)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(np.array(self.clip_confidences)[clip_correct],
                     label='Correct', alpha=0.7, bins=20, color='green')
        sns.histplot(np.array(self.clip_confidences)[~clip_correct],
                     label='Incorrect', alpha=0.7, bins=20, color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Clip-Level Confidence Distribution')
        plt.legend()

        # Chunk-level confidence distributions
        chunk_correct = np.array(self.chunk_true_labels) == np.array(self.chunk_predicted_labels)

        plt.subplot(1, 2, 2)
        sns.histplot(np.array(self.chunk_confidences)[chunk_correct],
                     label='Correct', alpha=0.7, bins=20, color='green')
        sns.histplot(np.array(self.chunk_confidences)[~chunk_correct],
                     label='Incorrect', alpha=0.7, bins=20, color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Chunk-Level Confidence Distribution')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confidence_distributions.png'))
        plt.close()

    def plot_example_analysis(self, num_examples=5):
        """Plot detailed analysis for a few example audio files"""
        audio_files, true_labels = self.get_audio_files_and_labels()

        # Select a small subset of files to analyze
        indices = np.random.choice(len(audio_files), min(num_examples, len(audio_files)), replace=False)

        for idx in indices:
            audio_file = audio_files[idx]
            true_label = true_labels[idx]

            self._analyze_single_file(audio_file, true_label)

    def _analyze_single_file(self, audio_file, true_label):
        """Generate detailed analysis visualizations for a single audio file"""
        audio, _ = load_audio(audio_file, self.sample_rate)

        # Initial hidden state (if model uses it)
        hidden_state = None
        if self.has_hidden_state:
            hidden_state = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)

        # For accumulating predictions
        all_chunk_probs = []  # Will store probabilities for all chunks
        all_chunk_times = []  # Will store start time of each chunk

        # For running statistics normalization
        running_mean, running_std = None, None

        # Process each chunk
        for chunk_start in range(0, len(audio), self.chunk_size_samples):
            chunk_end = min(chunk_start + self.chunk_size_samples, len(audio))
            chunk = audio[chunk_start:chunk_end]

            if len(chunk) < self.hop_length:  # Skip chunks that are too small
                continue

            # Store chunk start time
            chunk_time = chunk_start / self.sample_rate
            all_chunk_times.append(chunk_time)

            # Process chunk to get normalized spectrogram
            spec_input, running_mean, running_std = self.process_audio_chunk(
                chunk, running_mean, running_std
            )

            # Prepare inputs based on model requirements
            inputs = {'spectrogram_input': spec_input}
            output_names_to_fetch = ['output_logits']

            if self.has_hidden_state:
                inputs['hidden_state_input'] = hidden_state
                output_names_to_fetch.append('hidden_state_output')

            # Run inference
            outputs = self.session.run(output_names_to_fetch, inputs)
            logits = outputs[0]

            # Update hidden state if model uses it
            if self.has_hidden_state:
                hidden_state = outputs[1]

            # Extract predictions
            if isinstance(logits, np.ndarray):
                if logits.ndim == 3:  # (batch, seq, classes)
                    logits = logits[0, 0]
                elif logits.ndim == 2:  # (batch, classes)
                    logits = logits[0]

            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()

            all_chunk_probs.append(probabilities)

        # Convert to numpy array for easier indexing
        all_chunk_probs = np.array(all_chunk_probs)

        # Create visualization
        filename = os.path.basename(audio_file)
        plt.figure(figsize=(15, 10))

        # Plot waveform
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(len(audio))/self.sample_rate, audio)
        plt.title(f"Audio Waveform: {filename} (True: {self.class_names[true_label]})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # Plot class probabilities over time
        plt.subplot(3, 1, 2)
        for i in range(len(self.class_names)):
            class_probs = all_chunk_probs[:, i]
            plt.plot(all_chunk_times, class_probs, label=f"{self.class_names[i]}")
        plt.title("Class Probabilities by Chunk")
        plt.xlabel("Time (s)")
        plt.ylabel("Probability")
        plt.legend()
        plt.ylim(0, 1.05)

        # Plot top prediction and confidence
        plt.subplot(3, 1, 3)
        top_preds = np.argmax(all_chunk_probs, axis=1)
        top_confs = np.max(all_chunk_probs, axis=1)

        cmap = plt.cm.get_cmap('tab10', len(self.class_names))
        sc = plt.scatter(all_chunk_times, top_confs, c=top_preds, cmap=cmap,
                         vmin=-0.5, vmax=len(self.class_names)-0.5,
                         s=50, alpha=0.8)

        plt.colorbar(sc, ticks=range(len(self.class_names)),
                     label='Predicted Class')
        plt.title("Top Prediction Confidence by Chunk")
        plt.xlabel("Time (s)")
        plt.ylabel("Confidence")
        plt.ylim(0, 1.05)

        # Overall prediction for this file
        if len(all_chunk_probs) > 0:
            # Vote by class with highest average confidence
            class_avg_conf = {}
            for cls_idx in range(len(self.class_names)):
                avg_conf = np.mean(all_chunk_probs[:, cls_idx])
                class_avg_conf[cls_idx] = avg_conf

            clip_prediction = max(class_avg_conf.items(), key=lambda x: x[1])[0]

            plt.suptitle(f"File: {filename} | True: {self.class_names[true_label]} | "
                        f"Predicted: {self.class_names[clip_prediction]}",
                        fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Create output filename by replacing extension with png
        output_filename = os.path.splitext(os.path.basename(audio_file))[0] + "_analysis.png"
        plt.savefig(os.path.join(self.output_dir, output_filename))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate KWS model in streaming mode on test set")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset directory")
    parser.add_argument("--classes", type=str, required=True, help="Comma-separated list of class names")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size in milliseconds")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--n_mels", type=int, default=40, help="Number of mel bands")
    parser.add_argument("--n_fft", type=int, default=400, help="FFT size for Mel Spectrogram")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length for Mel Spectrogram")
    parser.add_argument("--running_stats", action="store_true", help="Use running statistics for normalization")
    parser.add_argument("--alpha", type=float, default=0.95, help="EMA coefficient for running statistics (0.0 to 1.0)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--analyze_examples", type=int, default=5, help="Number of example files to analyze in detail")

    args = parser.parse_args()

    # Parse class names
    class_names = args.classes.split(',')
    print(f"Classes: {class_names}")

    # Create evaluator
    evaluator = StreamingEvaluator(
        onnx_model_path=args.model,
        test_dir=args.test_dir,
        class_names=class_names,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        chunk_size_ms=args.chunk_size,
        use_running_stats=args.running_stats,
        alpha=args.alpha,
        output_dir=args.output_dir
    )

    # Evaluate test set
    evaluator.evaluate_test_set()

    # Compute and report metrics
    metrics = evaluator.compute_metrics()

    # Generate detailed analysis for a few examples
    if args.analyze_examples > 0:
        evaluator.plot_example_analysis(num_examples=args.analyze_examples)

    print(f"Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
