import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import time
import os
from dataloader import get_dataloader
from avsd_net import AIVECTOR_ConformerVEmbedding_SD_JOINT

def train_avsd():
    # Configuration
    CONFIG = {
        "input_dim": 40,
        "average_pooling": 3,
        "speaker_embedding_dim": 100,
        "output_speaker": 4,
        "audio_output_dim": 256,
        "video_embedding_dim": 256,
        "num_attention_heads": 4,
        "decoder_hidden_dim": 256,
        "dropout": 0.1,
    }
    
    # Training settings
    CSV_PATH = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    CHECKPOINT_DIR = "checkpoints"
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Initialize model
    print("🔄 Initializing model...")
    model = AIVECTOR_ConformerVEmbedding_SD_JOINT(CONFIG)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # DataLoader
    print("🔄 Loading data...")
    train_loader = get_dataloader(
        csv_path=CSV_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        max_speakers=CONFIG["output_speaker"],
        max_session_speakers=CONFIG["output_speaker"],
    )
    
    print(f"📊 Starting training with {len(train_loader)} batches per epoch")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['labels'].to(device)
            nframes = batch['nframes'].tolist()
            
            # Create dummy audio embeddings (as per your model)
            audio_embed = torch.zeros(audio.size(0), CONFIG["output_speaker"], 100).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(audio, audio_embed, video, nframes)
            
            # Compute loss for each speaker
            batch_loss = 0
            valid_speakers = 0
            
            for i, output in enumerate(outputs):
                if output is not None and len(output) > 0:
                    # Get ground truth for this speaker
                    gt_labels = labels[:, :, i].reshape(-1, 1)
                    
                    # Ensure same length
                    min_len = min(len(output), len(gt_labels))
                    if min_len > 0:
                        speaker_loss = criterion(output[:min_len], gt_labels[:min_len])
                        batch_loss += speaker_loss
                        valid_speakers += 1
            
            # Average loss across speakers
            if valid_speakers > 0:
                batch_loss = batch_loss / valid_speakers
                
                # Backward pass
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}' if valid_speakers > 0 else 'N/A',
                'Avg Loss': f'{(epoch_loss/batch_count):.4f}' if batch_count > 0 else 'N/A'
            })
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f'📈 Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': CONFIG
            }, checkpoint_path)
            print(f'💾 Checkpoint saved: {checkpoint_path}')
    
    print("✅ Training completed!")

if __name__ == "__main__":
    train_avsd()
