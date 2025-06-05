import os
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.rgbe_sequence_dataset import RGBESequenceDataset
from models.model import VitModel
from models.backbones.cnn import CNN_model
from models.losses.cross_entropy_loss import CrossEntropyLoss
import tqdm

def main(config_path, best_model_path, log_path):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. 构造 Dataset & DataLoader
    ds = cfg['dataset']
    train_ds = RGBESequenceDataset(
        data_root          = ds['train_dir'],
        window_size        = ds['window_size'],
        stride             = ds['stride'],
        enable_transform   = ds['enable_transform']
    )
    val_ds = RGBESequenceDataset(
        data_root          = ds['val_dir'],
        window_size        = ds['window_size'],
        stride             = ds['stride'],
        enable_transform   = ds['enable_transform']
    )
    test_ds = RGBESequenceDataset(
        data_root          = ds['test_dir'],
        window_size        = ds['window_size'],
        stride             = ds['stride'],
        enable_transform   = ds['enable_transform']
    )

    train_loader = DataLoader(
        train_ds,
        **cfg['train']
    )
    val_loader = DataLoader(
        val_ds,
        **cfg['val']
    )
    test_loader = DataLoader(
        test_ds,
        **cfg['test']
    )


    # 3. 构建模型、损失、优化器
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    # model = VitModel(cfg).to(device)
    model = CNN_model(cfg).to(device)

    loss_fn = nn.CrossEntropyLoss()
    # optim_cfg = cfg['optimizer']
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=optim_cfg['lr'],
    #     weight_decay=optim_cfg['weight_decay'],
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])


    best_acc = 0.0
    # 创建日志目录和文件
    if log_path is None:
        log_path = os.path.join(cfg['log_dir'], 'training_log_tmp.txt')
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['work_dir'], exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(" -----General Configuration------\n")
        f.write(f" Window_size: {ds['window_size']}\n")
        f.write(f" Stride: {ds['stride']}\n")
        f.write(f" Epochs: {cfg['epochs']}\n")
        f.write(f" Train batch size: {cfg['train']['batch_size']}\n")
        f.write(f" Validation batch size: {cfg['val']['batch_size']}\n")
        f.write(f" Test batch size: {cfg['test']['batch_size']}\n")
        f.write(f" ------ViT Model Configuration------\n")
        f.write(f" ViT Model: {cfg['model']}\n")
        f.write(f" ------CNN Model Configuration------\n")
        f.write(f" CNN Model: {cfg['cnn_model']}\n")
    for epoch in range(cfg['epochs']):
        # —— 训练 —— 
        train_start_time = time.time()
        print(f"[Epoch {epoch+1}/{cfg['epochs']}]:")
        print("Training...")
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm.tqdm(train_loader):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_end_time = time.time()

        print(f"Train Loss: {avg_train_loss:.4f}")
        with open(log_path, 'a') as f:
            f.write(f"\n[Epoch {epoch+1}/{cfg['epochs']}]\n")
            f.write(f"Train Loss: {avg_train_loss:.4f}\n")

        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        with open(log_path, 'a') as f:
            f.write(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}\n")
        
        num_train_batches = len(train_loader)
        num_train_samples = len(train_ds)
        print(f"Training statistics: {num_train_samples} samples in {num_train_batches} batches")
        with open(log_path, 'a') as f:
            f.write(f"Training statistics: {num_train_samples} samples in {num_train_batches} batches\n")
        
        # 记录训练时间
        train_time = train_end_time - train_start_time
        print(f"Training time: {train_time:.2f} seconds")
        with open(log_path, 'a') as f:
            f.write(f"Training time: {train_time:.2f} seconds\n")
        
        # —— 验证 —— 
        val_start_time = time.time()
        print("Validating...")
        model.eval()
        val_loss = correct = total = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm.tqdm(val_loader):
                imgs   = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                val_loss += loss_fn(logits, labels).item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_end_time = time.time()

        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        with open(log_path, 'a') as f:
            f.write(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        
        num_val_batches = len(val_loader)
        num_val_samples = len(val_ds)
        print(f"Validation statistics: {num_val_samples} samples in {num_val_batches} batches")
        with open(log_path, 'a') as f:
            f.write(f"Validation statistics: {num_val_samples} samples in {num_val_batches} batches\n")
        
        # 记录验证时间
        val_time = val_end_time - val_start_time
        print(f"Validation time: {val_time:.2f} seconds")
        with open(log_path, 'a') as f:
            f.write(f"Validation time: {val_time:.2f} seconds\n")

        # —— 保存最佳模型 —— 
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model.")
        print(f"-"*30)

    # —— 测试评估 —— 
    # Load the best model for testing
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")
    
    test_start_time = time.time()
    print("Testing...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(test_loader):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    test_acc = correct / total
    test_end_time = time.time()
    
    with open(log_path, 'a') as f:
        f.write(f"\nTest with best model from {best_model_path}\n")

    # 记录测试时间
    test_time = test_end_time - test_start_time
    print(f"Test time: {test_time:.2f} seconds")
    with open(log_path, 'a') as f:
        f.write(f"Test time: {test_time:.2f} seconds\n")

    print(f"Test Acc: {test_acc:.4f}")
    with open(log_path, 'a') as f:
        f.write(f"Test Acc: {test_acc:.4f} ({correct}/{total})\n")

    num_test_batches = len(test_loader)
    num_test_samples = len(test_ds)
    print(f"Test statistics: {num_test_samples} samples in {num_test_batches} batches")
    print("Training complete. Best validation accuracy:", best_acc)
    with open(log_path, 'a') as f:
        f.write(f"Test statistics: {num_test_samples} samples in {num_test_batches} batches\n")
        f.write(f"Training complete. Best validation accuracy: {best_acc}\n")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/qiangubuntu/research/har_rgbe/configs/har_rgbe.yaml',
                        help='Path to your_action_config.yaml')
    parser.add_argument('--model', type=str, default='results/checkpoints/vit_rgbe_2.pth',
                        help='Path to the pre-trained model')
    parser.add_argument('--log', type=str, default='/home/qiangubuntu/research/har_rgbe/results/logs/training_log_vit_rgbe_2.txt',
                        help='Path to the log file')
    args = parser.parse_args()
    main(args.config, args.model, args.log)
