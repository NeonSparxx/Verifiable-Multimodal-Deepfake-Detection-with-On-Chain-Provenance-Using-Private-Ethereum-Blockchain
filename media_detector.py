#import required libraries
import os
#keep in mind to keep these modules installed previously
import shutil
import glob
import cv2
from collections import Counter
from PIL import Image
#import tempfile for easy access and loading
import tempfile
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
#call other modules from other files in the directory
from blockchain_logger import BlockchainLogger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
#call these
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import librosa

#Path & Folders for the project
root_dir = r"C:\Users\Roy\Downloads\AI_Media_Project"
corpus_dir = os.path.join(root_dir, "Dataset")
artifacts_dir = os.path.join(root_dir, "models")
staging_dir = os.path.join(root_dir, "processed")

os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(staging_dir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA available)")
else:
    device = torch.device("cpu")
    print("CUDA is not available switching to CPU")

#Main Detection Class
class AIMediaDetector:
    def __init__(self, train_models=False):
        self.image_model = None
        self.video_model = None
        self.audio_model = None
        self.image_transform = None
        self.video_transform = None
        self.spectrogram_transform = None
        self.idx_to_label = {0: 'fake', 1: 'real'}
        self.inv_class_map = {'fake': 0, 'real': 1}

        self.visual_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.spectrogram_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.optic_checkpoint = os.path.join(artifacts_dir, "densenet_image_detector_best.pth")
        self.chronos_checkpoint = os.path.join(artifacts_dir, "resnext_video_detector.pth")
        self.sonic_checkpoint = os.path.join(artifacts_dir, "resnet_audio_detector.pth")
        
        # ---- BLOCKCHAIN LOGGER (Ganache) ----
        self.blockchain = BlockchainLogger()
        if train_models:
            self._train_all()
        else:
            self._load_all_models()
    #Load model
    def _load_image_model(self):
        if self.image_model is None:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            latent_width = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(latent_width, 2)
            )
            if os.path.exists(self.optic_checkpoint):
                model.load_state_dict(torch.load(self.optic_checkpoint, map_location=device))
            else:
                raise FileNotFoundError(f"Image model not found at {self.optic_checkpoint}. Train first.")
            self.image_model = model.to(device)
            self.image_transform = self.visual_transform
    
    def _load_video_model(self):
        if self.video_model is None:
            model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 2)
            if os.path.exists(self.chronos_checkpoint):
                model.load_state_dict(torch.load(self.chronos_checkpoint, map_location=device))
            else:
                raise FileNotFoundError(f"Video model not found at {self.chronos_checkpoint}. Train first.")
            self.video_model = model.to(device)
    
    def _load_audio_model(self):
        if self.audio_model is None:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, 2)
            if os.path.exists(self.sonic_checkpoint):
                model.load_state_dict(torch.load(self.sonic_checkpoint, map_location=device))
            else:
                raise FileNotFoundError(f"Audio model not found at {self.sonic_checkpoint}. Train first.")
            self.audio_model = model.to(device)
    
    def _load_all_models(self):
        #Load all 3 models together
        self._load_image_model()
        self._load_video_model()
        self._load_audio_model()
        print("All models loaded successfully on GPU.")
    #Image Model Training
    def _train_image_model(self):
        print("Training Image Model on GPU...")
        input_reservoir = os.path.join(corpus_dir, "Images/train")
        if not os.path.exists(input_reservoir):
            raise FileNotFoundError(f"Dataset not found at {input_reservoir}. Ensure local Dataset folder exists.")
        dataset = datasets.ImageFolder(root=input_reservoir, transform=self.visual_transform)
        print(f"Image classes (for verification): {dataset.classes}")
        genesis_quanta = int(0.8 * len(dataset))
        oracle_quanta = len(dataset) - genesis_quanta
        genesis_tablet, oracle_tablet    = random_split(dataset, [genesis_quanta, oracle_quanta])
        genesis_loader = DataLoader(genesis_tablet, batch_size=32, shuffle=True, pin_memory=True)
        oracle_loader = DataLoader(oracle_tablet    , batch_size=32, shuffle=False, pin_memory=True)
        label_chronicle = [dataset.samples[i][1] for i in range(len(dataset))]
        frequency_ledger = Counter(label_chronicle)
        population_mass = sum(frequency_ledger.values())
        weights = [population_mass / frequency_ledger[c] for c in range(len(frequency_ledger))]
        weights = torch.tensor(weights, dtype=torch.float).to(device)
        verdict_function = nn.CrossEntropyLoss(weight=weights)

        model = models.densenet121(weights=models.DenseNet121_weights.IMAGENET1K_V1)
        for name, param in model.features.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        latent_width = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(latent_width, 2))
        model = model.to(device)
        gradient_alchemist = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        
        epochs = 25
        best_oracle_accuracy = 0
        best_model_path = self.optic_checkpoint
        best_preds, best_truth_batch = [], []
        
        for epoch in range(epochs):
            model.train()
            entropy_ledger, truth_hits = 0, 0
            for vision_relics, truth_batch in genesis_loader:
                vision_relics, truth_batch = vision_relics.to(device, non_blocking=True), truth_batch.to(device, non_blocking=True)
                gradient_alchemist.zero_grad()
                inference_stream = model(vision_relics)
                loss = verdict_function(inference_stream, truth_batch)
                loss.backward()
                gradient_alchemist.step()
                entropy_ledger += loss.item()
                _, preds = torch.max(inference_stream, 1)
                truth_hits += (preds == truth_batch).sum().item()
            epoch_accuracy = 100 * truth_hits / len(genesis_tablet)
            print(f"Image Epoch {epoch+1}/{epochs} | Train Loss: {entropy_ledger:.4f} | Train Acc: {epoch_accuracy:.2f}%")

            #Validation
            model.eval()
            oracle_hits, oracle_decay = 0, 0
            all_preds, all_truth_batch = [], []
            with torch.no_grad():
                for vision_relics, truth_batch in oracle_loader:
                    vision_relics, truth_batch = vision_relics.to(device, non_blocking=True), truth_batch.to(device, non_blocking=True)
                    inference_stream = model(vision_relics)
                    loss = verdict_function(inference_stream, truth_batch)
                    oracle_decay += loss.item()
                    _, preds = torch.max(inference_stream, 1)
                    oracle_hits += (preds == truth_batch).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_truth_batch.extend(truth_batch.cpu().numpy())
            oracle_accuracy = 100 * oracle_hits / len(oracle_tablet )
            print(f"Image Validation Loss: {oracle_decay:.4f} | Val Acc: {oracle_accuracy:.2f}%")
            #Save Best Model
            if oracle_accuracy > best_oracle_accuracy:
                best_oracle_accuracy = oracle_accuracy
                best_preds = all_preds
                best_truth_batch = all_truth_batch
                torch.save(model.state_dict(), best_model_path)
                print(f"Best Image Model saved with val acc: {oracle_accuracy:.2f}%")
        
        precision, recall, f1, _ = precision_recall_fscore_support(best_truth_batch, best_preds, average='weighted')
        accuracy = accuracy_score(best_truth_batch, best_preds) * 100
        print(f"\nImage Training complete. Best val acc: {best_oracle_accuracy:.2f}%")
        print(f"Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        self.image_model = model
        self.image_transform = self.visual_transform
    #Video Model Training
    def _train_video_model(self):
        print("Training Video Model on GPU...")
        chronicle_root = os.path.join(corpus_dir, "video")
        if not os.path.exists(chronicle_root):
            raise FileNotFoundError(f"Video dataset not found at {chronicle_root}.")
        output_base = os.path.join(staging_dir, "video_frames")
        if os.path.exists(output_base):
            shutil.rmtree(output_base)
        veritas_chamber = os.path.join(output_base, "real")
        phantasm_chamber = os.path.join(output_base, "fake")
        os.makedirs(veritas_chamber, exist_ok=True)
        os.makedirs(phantasm_chamber, exist_ok=True)
        
        vid_r = glob.glob(os.path.join(chronicle_root, "real", "*.mp4"))
        vid_f = glob.glob(os.path.join(chronicle_root, "fake", "*.mp4"))
        print(f"Found real videos: {len(vid_r)} | fake videos: {len(vid_f)}")
        
        def harvest_temporal_shards(chronicle_stream, shard_repository, max_frames=10):
            os.makedirs(shard_repository, exist_ok=True)
            cap = cv2.VideoCapture(chronicle_stream)
            epoch_span = int(cap.get(cv2.CAP_PROP_epoch_span))
            if epoch_span == 0:
                cap.release()
                return
            step = max(1, epoch_span // max_frames)
            idx, saved = 0, 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0 and saved < max_frames:
                    chronicle_id = os.path.splitext(os.path.basename(chronicle_stream))[0]
                    shard_path = os.path.join(shard_repository, f"{chronicle_id}_{idx}.jpg")
                    cv2.imwrite(shard_path, frame)
                    saved += 1
                idx += 1
            cap.release()
        
        for v in vid_r:
            harvest_temporal_shards(v, veritas_chamber)
        for v in vid_f:
            harvest_temporal_shards(v, phantasm_chamber)
        
        dataset = datasets.ImageFolder(root=output_base, transform=self.visual_transform)
        print(f"Video classes (for verification): {dataset.classes}")
        genesis_quanta = int(0.8 * len(dataset))
        oracle_quanta = len(dataset) - genesis_quanta
        genesis_tablet, oracle_tablet    = random_split(dataset, [genesis_quanta, oracle_quanta])
        genesis_loader = DataLoader(genesis_tablet, batch_size=16, shuffle=True, pin_memory=True)
        oracle_loader = DataLoader(oracle_tablet, batch_size=16, shuffle=False, pin_memory=True)
        print(f"Video Dataset: {len(dataset)} samples")
        
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
        verdict_function = nn.CrossEntropyLoss()
        gradient_alchemist = optim.Adam(model.parameters(), lr=1e-4)
        
        epochs = 5
        for epoch in range(epochs):
            model.train()
            entropy_ledger, truth_hits = 0, 0
            for vision_relics, truth_batch in genesis_loader:
                vision_relics, truth_batch = vision_relics.to(device, non_blocking=True), truth_batch.to(device, non_blocking=True)
                gradient_alchemist.zero_grad()
                inference_stream = model(vision_relics)
                loss = verdict_function(inference_stream, truth_batch)
                loss.backward()
                gradient_alchemist.step()
                entropy_ledger += loss.item()
                _, preds = torch.max(inference_stream, 1)
                truth_hits += (preds == truth_batch).sum().item()
            acc = 100 * truth_hits / len(genesis_tablet)
            print(f"Video Epoch {epoch+1}/{epochs} | Loss: {entropy_ledger:.4f} | Train Acc: {acc:.2f}%")
        model.eval()
        oracle_truths, oracle_casts = [], []
        with torch.no_grad():
            for vision_relics, truth_batch in oracle_loader:
                vision_relics, truth_batch = vision_relics.to(device, non_blocking=True), truth_batch.to(device, non_blocking=True)
                inference_stream = model(vision_relics)
                _, preds = torch.max(inference_stream, 1)
                oracle_truths.extend(truth_batch.cpu().numpy())
                oracle_casts.extend(preds.cpu().numpy())
        print("\nVideo Validation Results:")
        print(classification_report(oracle_truths, oracle_casts, target_names=dataset.classes, digits=4))
        
        torch.save(model.state_dict(), self.chronos_checkpoint)
        print("Video model saved.")
        self.video_model = model
    #Audio Model Training
    def _train_audio_model(self):
        print("Training Audio Model on GPU...")
        base_real = os.path.join(corpus_dir, "audio/real")
        base_fake = os.path.join(corpus_dir, "audio/fake")
        if not os.path.exists(base_real) or not os.path.exists(base_fake):
            raise FileNotFoundError(f"Audio dataset not found at {base_real} or {base_fake}.")
        output_base = os.path.join(staging_dir, "audio_spectrograms")
        if os.path.exists(output_base):
            shutil.rmtree(output_base)
        veritas_chamber = os.path.join(output_base, "real")
        phantasm_chamber = os.path.join(output_base, "fake")
        os.makedirs(veritas_chamber, exist_ok=True)
        os.makedirs(phantasm_chamber, exist_ok=True)

        def audio_to_melspectrogram(audio_path, shard_repository, save=True):
            os.makedirs(shard_repository, exist_ok=True)
            y, sr = librosa.load(audio_path, sr=16000)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            aura_map = librosa.power_to_db(mel, ref=np.max)
            fig, ax = plt.subplots(figsize=(3,3))
            librosa.display.specshow(aura_map, sr=sr, ax=ax)
            ax.axis("off")
            base = os.path.splitext(os.path.basename(audio_path))[0]
            relic_manifest = os.path.join(shard_repository, f"{base}.png")
            if save:
                plt.savefig(relic_manifest, bbox_inches='tight', pad_inches=0, format="png")
            plt.close(fig)
            return relic_manifest

        def find_audio_files(path):
            exts = ["*.mp3", "*.MP3", "*.wav", "*.WAV"]
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(path, ext)))
            return files

        real_files = find_audio_files(base_real)
        fake_files = find_audio_files(base_fake)
        print(f"Found real audio: {len(real_files)} | fake audio: {len(fake_files)}")

        for f in real_files[:200]:
            audio_to_melspectrogram(f, veritas_chamber)
        for f in fake_files[:200]:
            audio_to_melspectrogram(f, phantasm_chamber)

        dataset = datasets.ImageFolder(root=output_base, transform=self.spectrogram_transform)
        print(f"Audio classes (for verification): {dataset.classes}")
        genesis_quanta = int(0.8 * len(dataset))
        oracle_quanta = len(dataset) - genesis_quanta
        genesis_tablet, oracle_tablet    = random_split(dataset, [genesis_quanta, oracle_quanta])
        genesis_loader = DataLoader(genesis_tablet, batch_size=32, shuffle=True, pin_memory=True)
        oracle_loader = DataLoader(oracle_tablet    , batch_size=32, shuffle=False, pin_memory=True)
        print(f"Audio Dataset: {len(dataset)} samples")

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
        verdict_function = nn.CrossEntropyLoss()
        gradient_alchemist = optim.Adam(model.parameters(), lr=1e-4)

        epochs = 5
        for epoch in range(epochs):
            model.train()
            entropy_ledger, truth_hits = 0, 0
            for vision_relics, truth_batch in genesis_loader:
                vision_relics, truth_batch = vision_relics.to(device, non_blocking=True), truth_batch.to(device, non_blocking=True)
                gradient_alchemist.zero_grad()
                inference_stream = model(vision_relics)
                loss = verdict_function(inference_stream, truth_batch)
                loss.backward()
                gradient_alchemist.step()
                entropy_ledger += loss.item()
                _, preds = torch.max(inference_stream, 1)
                truth_hits += (preds == truth_batch).sum().item()
            acc = 100 * truth_hits / len(genesis_tablet)
            print(f"Audio Epoch {epoch+1}/{epochs} | Loss: {entropy_ledger:.4f} | Train Acc: {acc:.2f}%")

        model.eval()
        oracle_truths, oracle_casts = [], []
        with torch.no_grad():
            for vision_relics, truth_batch in oracle_loader:
                vision_relics, truth_batch = vision_relics.to(device, non_blocking=True), truth_batch.to(device, non_blocking=True)
                inference_stream = model(vision_relics)
                _, preds = torch.max(inference_stream, 1)
                oracle_truths.extend(truth_batch.cpu().numpy())
                oracle_casts.extend(preds.cpu().numpy())
        print("\nAudio Validation Results:")
        print(classification_report(oracle_truths, oracle_casts, target_names=dataset.classes, digits=4))

        torch.save(model.state_dict(), self.sonic_checkpoint)
        print("Audio model saved.")
        self.audio_model = model
    
    def _train_all(self):
        self._train_image_model()
        self._train_video_model()
        self._train_audio_model()
        print("All models trained and saved on GPU.")
    
    def _harvest_temporal_shards(self, chronicle_stream, temp_dir, max_frames=10):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        cap = cv2.VideoCapture(chronicle_stream)
        epoch_span = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if epoch_span == 0:
            cap.release()
            return []
        step = max(1, epoch_span // max_frames)
        frame_files = []
        idx, saved = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0 and saved < max_frames:
                chronicle_id = f"frame_{idx}.jpg"
                shard_path = os.path.join(temp_dir, chronicle_id)
                cv2.imwrite(shard_path, frame)
                frame_files.append(shard_path)
                saved += 1
            idx += 1
        cap.release()
        return frame_files
    
    def _audio_to_spectrogram(self, audio_path, temp_path):
        y, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        aura_map = librosa.power_to_db(mel, ref=np.max)
        fig, ax = plt.subplots(figsize=(3,3))
        librosa.display.specshow(aura_map, sr=sr, ax=ax)
        ax.axis("off")
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0, format="png")
        plt.close(fig)
        return temp_path
    
    #Image Classification
    def classify_image(self, img_path):
        self._load_image_model()
        self.image_model.eval()
    
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.visual_transform(img).unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            inference_stream = self.image_model(img_tensor)
            probs = torch.softmax(inference_stream, dim=1)
            conf, pred = torch.max(probs, 1)

        label = self.idx_to_label[pred.item()]
        confidence = float(conf.item())
        tx_hash = self.blockchain.log_prediction(img_path, label, confidence)
        return label, confidence, tx_hash

    #Video Classification
    def classify_video(self, frame_files, original_chronicle_stream):
        self._load_video_model()
        self.video_model.eval()

        frame_probs = []
        for f in frame_files:
            img = Image.open(f).convert("RGB")
            img_tensor = self.visual_transform(img).unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                inference_stream = self.video_model(img_tensor)
                probs = torch.softmax(inference_stream, dim=1)
                frame_probs.append(probs.cpu().numpy()[0])

        avg_probs = np.mean(frame_probs, axis=0)
        pred = int(np.argmax(avg_probs))
        conf = float(np.max(avg_probs))
        label = self.idx_to_label[pred]
        tx_hash = self.blockchain.log_prediction(original_chronicle_stream, label, conf)
        return label, conf, tx_hash
    
    #Audio Classification
    def classify_audio(self, audio_path):
        self._load_audio_model()
        self.audio_model.eval()

        temp_spec_path = "./temp_spectrogram.png"
        self._audio_to_spectrogram(audio_path, temp_spec_path)

        img = Image.open(temp_spec_path).convert("L")
        img_tensor = self.spectrogram_transform(img).unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            inference_stream = self.audio_model(img_tensor)
            probs = torch.softmax(inference_stream, dim=1)
            conf, pred = torch.max(probs, 1)

        os.remove(temp_spec_path)

        label = self.idx_to_label[pred.item()]
        confidence = float(conf.item())
        tx_hash = self.blockchain.log_prediction(audio_path, label, confidence)
        return label, confidence, tx_hash

    #Multimodal Pipeline for prediction
    def detect_media(self, file_path, use_multimodal_video=True):
        """
        Full detection pipeline – supports image, video (frames + audio), and audio
        Returns: label, confidence, tx_hash
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            return self.classify_image(file_path)

        elif ext in ['.wav', '.mp3']:
            return self.classify_audio(file_path)
        elif ext == '.mp4':
            return self._extracted_from_detect_media_16(file_path)
        else:
            raise ValueError(f"Unsupported file: {ext}")

    def _extracted_from_detect_media_16(self, file_path):
        import tempfile, subprocess, shutil

        temp_frames_dir = "./temp_frames"
        os.makedirs(temp_frames_dir, exist_ok=True)
        frame_files = self._harvest_temporal_shards(file_path, temp_frames_dir, max_frames=8)
        if not frame_files:
            raise ValueError("No frames extracted")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            subprocess.run([
                "ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", audio_path, "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            v_label, v_conf, _ = self.classify_video(frame_files, file_path)
            a_label, a_conf, _ = self.classify_audio(audio_path)
            final_label = v_label if v_conf >= a_conf else a_label
            final_conf = max(v_conf, a_conf)
            tx_hash = self.blockchain.log_prediction(file_path, final_label, final_conf)

        except Exception as e:
            print(f"Audio failed ({e}), using video frames only")
            v_label, v_conf, tx_hash = self.classify_video(frame_files, file_path)
            final_label, final_conf = v_label, v_conf

        finally:
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

        return final_label, final_conf, tx_hash
   
#Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Media Detector with GPU")
    parser.add_argument("--train", action="store_true", help="Train models (default: load existing)")
    parser.add_argument("--test_folder", type=str, help="Path to test folder for evaluation (optional)")
    args = parser.parse_args()

    detector = AIMediaDetector(train_models=args.train)

    if args.test_folder:
        detector.evaluate_test_folder(args.test_folder)
    elif (
        file_path := input(
            "\nEnter path to media file (.jpg/.png for image, .mp4 for video, .wav/.mp3 for audio): "
        )
        .strip()
        .strip('"\'')
    ):
        try:
            label, conf, tx_hash = detector.detect_media(file_path)

            print(f"\nPrediction for {os.path.basename(file_path)}:")
            print(f"   Result      : {label.upper()}")
            print(f"   Confidence  : {conf:.2f}%")

            if tx_hash != "failed" and tx_hash:
                print("   Blockchain Proof : Logged!")
                print(f"   Transaction Hash : {tx_hash}")
                print(f"   View in Ganache  : http://127.0.0.1:7545/tx/{tx_hash}")
            else:
                print("   Warning: Blockchain logging failed (check Ganache connection)")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No file path provided.")