import os
import shutil
import glob
import cv2
from collections import Counter
from PIL import Image
import tempfile
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from blockchain_logger import BlockchainLogger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import librosa

base_project_dir = r"C:\Users\Roy\Downloads\AI_Media_Project"
dataset_dir = os.path.join(base_project_dir, "Dataset")
models_dir = os.path.join(base_project_dir, "models")
processed_dir = os.path.join(base_project_dir, "processed")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA available)")
else:
    device = torch.device("cpu")
    print("CUDA not available; using CPU")

class AIMediaDetector:
    def __init__(self, train_models=False):
        self.image_model = None
        self.video_model = None
        self.audio_model = None
        self.image_transform = None
        self.video_transform = None
        self.audio_transform = None
        self.class_to_idx = {0: 'fake', 1: 'real'}
        self.inv_class_map = {'fake': 0, 'real': 1}

        self.frame_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.audio_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.image_model_path = os.path.join(models_dir, "densenet_image_detector_best.pth")
        self.video_model_path = os.path.join(models_dir, "resnext_video_detector.pth")
        self.audio_model_path = os.path.join(models_dir, "resnet_audio_detector.pth")
        
        # ---- BLOCKCHAIN LOGGER (Ganache) ----
        self.blockchain = BlockchainLogger()
        if train_models:
            self._train_all()
        else:
            self._load_all_models()
    
    def _load_image_model(self):
        if self.image_model is None:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            num_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 2)
            )
            if os.path.exists(self.image_model_path):
                model.load_state_dict(torch.load(self.image_model_path, map_location=device))
            else:
                raise FileNotFoundError(f"Image model not found at {self.image_model_path}. Train first.")
            self.image_model = model.to(device)
            self.image_transform = self.frame_transform
    
    def _load_video_model(self):
        if self.video_model is None:
            model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 2)
            if os.path.exists(self.video_model_path):
                model.load_state_dict(torch.load(self.video_model_path, map_location=device))
            else:
                raise FileNotFoundError(f"Video model not found at {self.video_model_path}. Train first.")
            self.video_model = model.to(device)
    
    def _load_audio_model(self):
        if self.audio_model is None:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, 2)
            if os.path.exists(self.audio_model_path):
                model.load_state_dict(torch.load(self.audio_model_path, map_location=device))
            else:
                raise FileNotFoundError(f"Audio model not found at {self.audio_model_path}. Train first.")
            self.audio_model = model.to(device)
    
    def _load_all_models(self):
        self._load_image_model()
        self._load_video_model()
        self._load_audio_model()
        print("All models loaded successfully on GPU.")
    
    def _train_image_model(self):
        print("Training Image Model on GPU...")
        data_dir = os.path.join(dataset_dir, "Images/train")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset not found at {data_dir}. Ensure local Dataset folder exists.")
        dataset = datasets.ImageFolder(root=data_dir, transform=self.frame_transform)
        print(f"Image classes (for verification): {dataset.classes}")
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)
        targets = [dataset.samples[i][1] for i in range(len(dataset))]
        counts = Counter(targets)
        total = sum(counts.values())
        weights = [total / counts[c] for c in range(len(counts))]
        weights = torch.tensor(weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for name, param in model.features.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 2))
        model = model.to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        
        epochs = 25
        best_val_acc = 0
        best_model_path = self.image_model_path
        best_preds, best_labels = [], []
        
        for epoch in range(epochs):
            model.train()
            running_loss, correct = 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
            train_acc = 100 * correct / len(train_ds)
            print(f"Image Epoch {epoch+1}/{epochs} | Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")
            model.eval()
            val_correct, val_loss = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            val_acc = 100 * val_correct / len(val_ds)
            print(f"Image Validation Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_preds = all_preds
                best_labels = all_labels
                torch.save(model.state_dict(), best_model_path)
                print(f"Best Image Model saved with val acc: {val_acc:.2f}%")
        
        precision, recall, f1, _ = precision_recall_fscore_support(best_labels, best_preds, average='weighted')
        accuracy = accuracy_score(best_labels, best_preds) * 100
        print(f"\nImage Training complete. Best val acc: {best_val_acc:.2f}%")
        print(f"Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        self.image_model = model
        self.image_transform = self.frame_transform
    
    def _train_video_model(self):
        print("Training Video Model on GPU...")
        base_dir = os.path.join(dataset_dir, "video")
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Video dataset not found at {base_dir}.")
        output_base = os.path.join(processed_dir, "video_frames")
        if os.path.exists(output_base):
            shutil.rmtree(output_base)
        real_dir = os.path.join(output_base, "real")
        fake_dir = os.path.join(output_base, "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        real_videos = glob.glob(os.path.join(base_dir, "real", "*.mp4"))
        fake_videos = glob.glob(os.path.join(base_dir, "fake", "*.mp4"))
        print(f"Found real videos: {len(real_videos)} | fake videos: {len(fake_videos)}")
        
        def extract_frames(video_path, output_dir, max_frames=10):
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                cap.release()
                return
            step = max(1, frame_count // max_frames)
            idx, saved = 0, 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0 and saved < max_frames:
                    frame_name = os.path.splitext(os.path.basename(video_path))[0]
                    frame_path = os.path.join(output_dir, f"{frame_name}_{idx}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved += 1
                idx += 1
            cap.release()
        
        for v in real_videos:
            extract_frames(v, real_dir)
        for v in fake_videos:
            extract_frames(v, fake_dir)
        
        dataset = datasets.ImageFolder(root=output_base, transform=self.frame_transform)
        print(f"Video classes (for verification): {dataset.classes}")
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, pin_memory=True)
        print(f"Video Dataset: {len(dataset)} samples")
        
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        epochs = 5
        for epoch in range(epochs):
            model.train()
            running_loss, correct = 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
            acc = 100 * correct / len(train_ds)
            print(f"Video Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Train Acc: {acc:.2f}%")
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        print("\nVideo Validation Results:")
        print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))
        
        torch.save(model.state_dict(), self.video_model_path)
        print("Video model saved.")
        self.video_model = model
    
    def _train_audio_model(self):
        print("Training Audio Model on GPU...")
        base_real = os.path.join(dataset_dir, "audio/real")
        base_fake = os.path.join(dataset_dir, "audio/fake")
        if not os.path.exists(base_real) or not os.path.exists(base_fake):
            raise FileNotFoundError(f"Audio dataset not found at {base_real} or {base_fake}.")
        output_base = os.path.join(processed_dir, "audio_spectrograms")
        if os.path.exists(output_base):
            shutil.rmtree(output_base)
        real_dir = os.path.join(output_base, "real")
        fake_dir = os.path.join(output_base, "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        
        def audio_to_melspectrogram(audio_path, output_dir, save=True):
            os.makedirs(output_dir, exist_ok=True)
            y, sr = librosa.load(audio_path, sr=16000)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            fig, ax = plt.subplots(figsize=(3,3))
            librosa.display.specshow(mel_db, sr=sr, ax=ax)
            ax.axis("off")
            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_file = os.path.join(output_dir, base + ".png")
            if save:
                plt.savefig(out_file, bbox_inches='tight', pad_inches=0, format="png")
            plt.close(fig)
            return out_file
        
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
            audio_to_melspectrogram(f, real_dir)
        for f in fake_files[:200]:
            audio_to_melspectrogram(f, fake_dir)
        
        dataset = datasets.ImageFolder(root=output_base, transform=self.audio_transform)
        print(f"Audio classes (for verification): {dataset.classes}")
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)
        print(f"Audio Dataset: {len(dataset)} samples")
        
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        epochs = 5
        for epoch in range(epochs):
            model.train()
            running_loss, correct = 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
            acc = 100 * correct / len(train_ds)
            print(f"Audio Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Train Acc: {acc:.2f}%")

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        print("\nAudio Validation Results:")
        print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))
        
        torch.save(model.state_dict(), self.audio_model_path)
        print("Audio model saved.")
        self.audio_model = model
    
    def _train_all(self):
        self._train_image_model()
        self._train_video_model()
        self._train_audio_model()
        print("All models trained and saved on GPU.")
    
    def _extract_frames(self, video_path, temp_dir, max_frames=10):
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            return []
        step = max(1, frame_count // max_frames)
        frame_files = []
        idx, saved = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0 and saved < max_frames:
                frame_name = f"frame_{idx}.jpg"
                frame_path = os.path.join(temp_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_files.append(frame_path)
                saved += 1
            idx += 1
        cap.release()
        return frame_files
    
    def _audio_to_spectrogram(self, audio_path, temp_path):
        y, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig, ax = plt.subplots(figsize=(3,3))
        librosa.display.specshow(mel_db, sr=sr, ax=ax)
        ax.axis("off")
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0, format="png")
        plt.close(fig)
        return temp_path
    
    def _predict_image(self, img_path):
        self._load_image_model()
        self.image_model.eval()
    
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.frame_transform(img).unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            outputs = self.image_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = self.class_to_idx[pred.item()]
        confidence = float(conf.item())
        tx_hash = self.blockchain.log_prediction(img_path, label, confidence)
        return label, confidence, tx_hash

    def _predict_video_frames(self, frame_files, original_video_path):
        self._load_video_model()
        self.video_model.eval()

        frame_probs = []
        for f in frame_files:
            img = Image.open(f).convert("RGB")
            img_tensor = self.frame_transform(img).unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                outputs = self.video_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                frame_probs.append(probs.cpu().numpy()[0])

        avg_probs = np.mean(frame_probs, axis=0)
        pred = int(np.argmax(avg_probs))
        conf = float(np.max(avg_probs))
        label = self.class_to_idx[pred]
        tx_hash = self.blockchain.log_prediction(original_video_path, label, conf)
        return label, conf, tx_hash
    
    def _predict_audio(self, audio_path):
        self._load_audio_model()
        self.audio_model.eval()

        temp_spec_path = "./temp_spectrogram.png"
        self._audio_to_spectrogram(audio_path, temp_spec_path)

        img = Image.open(temp_spec_path).convert("L")
        img_tensor = self.audio_transform(img).unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            outputs = self.audio_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        os.remove(temp_spec_path)

        label = self.class_to_idx[pred.item()]
        confidence = float(conf.item())
        tx_hash = self.blockchain.log_prediction(audio_path, label, confidence)
        return label, confidence, tx_hash


    def detect_media(self, file_path, use_multimodal_video=True):
        """
        Full detection pipeline – supports image, video (frames + audio), and audio
        Returns: label, confidence, tx_hash
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            return self._predict_image(file_path)
        
        elif ext in ['.wav', '.mp3']:
            return self._predict_audio(file_path)
        elif ext == '.mp4':
            import tempfile, subprocess, shutil
            
            temp_frames_dir = "./temp_frames"
            os.makedirs(temp_frames_dir, exist_ok=True)
            frame_files = self._extract_frames(file_path, temp_frames_dir, max_frames=8)
            if not frame_files:
                raise ValueError("No frames extracted")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = f.name
            
            try:
                subprocess.run([
                    "ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", audio_path, "-y"
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                v_label, v_conf, _ = self._predict_video_frames(frame_files, file_path)
                a_label, a_conf, _ = self._predict_audio(audio_path)
                final_label = v_label if v_conf >= a_conf else a_label
                final_conf = max(v_conf, a_conf)
                tx_hash = self.blockchain.log_prediction(file_path, final_label, final_conf)
                
            except Exception as e:
                print(f"Audio failed ({e}), using video frames only")
                v_label, v_conf, tx_hash = self._predict_video_frames(frame_files, file_path)
                final_label, final_conf = v_label, v_conf
            
            finally:
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            
            return final_label, final_conf, tx_hash
        
        else:
            raise ValueError(f"Unsupported file: {ext}")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Media Detector with GPU")
    parser.add_argument("--train", action="store_true", help="Train models (default: load existing)")
    parser.add_argument("--test_folder", type=str, help="Path to test folder for evaluation (optional)")
    args = parser.parse_args()
    
    detector = AIMediaDetector(train_models=args.train)
    
    if args.test_folder:
        detector.evaluate_test_folder(args.test_folder)
    else:
        file_path = input("\nEnter path to media file (.jpg/.png for image, .mp4 for video, .wav/.mp3 for audio): ").strip().strip('"\'')
        if file_path:
            try:
                label, conf, tx_hash = detector.detect_media(file_path)
                
                print(f"\nPrediction for {os.path.basename(file_path)}:")
                print(f"   Result      : {label.upper()}")
                print(f"   Confidence  : {conf:.2f}%")
                
                if tx_hash != "failed" and tx_hash:
                    print(f"   Blockchain Proof : Logged!")
                    print(f"   Transaction Hash : {tx_hash}")
                    print(f"   View in Ganache  : http://127.0.0.1:7545/tx/{tx_hash}")
                else:
                    print("   Warning: Blockchain logging failed (check Ganache connection)")
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No file path provided.")