import os
import glob
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
from models.EVSSM import EVSSM

def tile_inference(model, img, tile_size=1024, overlap=128, fp16=False):
    """
    分塊推理 (Tiled Inference) 函數，用於處理大圖以防 VRAM 爆炸 (OOM)。
    img: torch.Tensor, shape (1, C, H, W)
    """
    b, c, h, w = img.shape
    assert b == 1
    
    output = torch.zeros_like(img)
    weight_mask = torch.zeros((1, 1, h, w), device=img.device)
    
    # 計算滑動窗口步長 (步長 = 塊大小 - 重疊大小)
    stride_h = tile_size - overlap
    stride_w = tile_size - overlap
    
    # 產生水平與垂直切片的起點
    h_starts = list(range(0, h - tile_size, stride_h)) + [max(0, h - tile_size)]
    w_starts = list(range(0, w - tile_size, stride_w)) + [max(0, w - tile_size)]
    # 去除重複的起點
    h_starts = sorted(list(set(h_starts)))
    w_starts = sorted(list(set(w_starts)))
    
    total_tiles = len(h_starts) * len(w_starts)
    
    # 使用 tqdm 顯示內部切片處理進度
    pbar = tqdm(total=total_tiles, desc="  分塊處理中", leave=False)
    for hs in h_starts:
        for ws in w_starts:
            # 裁剪當前分塊
            tile = img[:, :, hs:hs+tile_size, ws:ws+tile_size]
            
            # 模型推理 (使用半精度與推理模式優化速度)
            with torch.inference_mode(), torch.amp.autocast('cuda', enabled=fp16):
                tile_pred = model(tile)
                # 確保返回的是 float32 以便後續權重計算
                tile_pred = tile_pred.float()
            
            # 建立線性漸變的權重遮罩，防止邊界縫隙
            th, tw = tile.shape[2], tile.shape[3]
            mask_h = torch.ones(th, device=img.device)
            mask_w = torch.ones(tw, device=img.device)
            
            # 若不是影像最外側邊界，則在重疊區域套用漸變
            if hs > 0:
                mask_h[:overlap] = torch.linspace(0, 1, overlap, device=img.device)
            if hs + th < h:
                mask_h[-overlap:] = torch.linspace(1, 0, overlap, device=img.device)
            if ws > 0:
                mask_w[:overlap] = torch.linspace(0, 1, overlap, device=img.device)
            if ws + tw < w:
                mask_w[-overlap:] = torch.linspace(1, 0, overlap, device=img.device)
                
            tile_mask = mask_h.unsqueeze(1) * mask_w.unsqueeze(0) # (th, tw)
            tile_mask = tile_mask.unsqueeze(0).unsqueeze(0) # (1, 1, th, tw)
            
            # 累加結果與權重
            output[:, :, hs:hs+th, ws:ws+tw] += tile_pred * tile_mask
            weight_mask[:, :, hs:hs+th, ws:ws+tw] += tile_mask
            pbar.update(1)
            
    pbar.close()
    # 除以總權重以平滑融合重疊區域
    output /= (weight_mask + 1e-8)
    return output

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EVSSM 大圖推理腳本")
    parser.add_argument('--input_dir', type=str, default='./my_inputs/', help='輸入模糊影像資料夾')
    parser.add_argument('--output_dir', type=str, default='./my_outputs/', help='輸出去模糊影像資料夾')
    parser.add_argument('--model_path', type=str, required=True, help='預訓練模型路徑 (.pth)')
    parser.add_argument('--tile_size', type=int, default=1536, help='分塊大小 (T4 GPU 推薦 1024 或 1536)')
    parser.add_argument('--overlap', type=int, default=128, help='分塊重疊像素大小')
    parser.add_argument('--no_fp16', action='store_true', help='停用半精度 (FP16)，改用無損單精度 (FP32) 進行推理，可防止高光區域產生純色色塊')
    parser.add_argument('--num_images', type=int, default=-1, help='限制只處理前 N 張影像，設為 -1表示處理所有影像')
    parser.add_argument('--resize', type=int, default=-1, help='將影像長邊縮放到指定大小（如 1280 或 1600），-1 表示不縮放。超大圖推薦縮放以獲得最佳效果與速度')
    parser.add_argument('--iters', type=int, default=1, help='遞迴去模糊次數。對於極大晃動可試試 2 或 3 次，但可能會使影像變平滑')
    args = parser.parse_args()
    
    fp16_enabled = not args.no_fp16

    # 建立輸出資料夾
    os.makedirs(args.output_dir, exist_ok=True)

    # 載入模型
    print("正在載入 EVSSM 模型...")
    model = EVSSM()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"目前使用的硬體裝置為: {device}")
    if device.type == 'cpu':
        print("⚠️ 警告：目前沒有偵測到 GPU，正在使用 CPU 進行推理，速度將會非常緩慢！")
        print("💡 提示：請確保已在 Kaggle/Colab 的 Notebook 中啟用 GPU 加速器。")
    model = model.to(device)
    
    # 載入權重
    print(f"正在載入權重: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    if 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 搜尋輸入圖片
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    if not img_paths:
        print(f"錯誤：在 {args.input_dir} 中找不到任何影像檔案！請確認路徑或上傳影像。")
        return
    
    # 限制處理圖片的數量，方便快速測試
    if args.num_images > 0:
        img_paths = img_paths[:args.num_images]
        print(f"依設定只處理前 {args.num_images} 張影像，開始去模糊推理...")
    else:
        print(f"找到 {len(img_paths)} 張影像，開始去模糊推理...")

    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)
        
        # 讀取圖片並進行等比例縮放 (解決超高解析度下模糊跨度超出感受野的問題)
        img_pil = Image.open(img_path).convert('RGB')
        if args.resize > 0:
            w, h = img_pil.size
            if max(w, h) > args.resize:
                scale = args.resize / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                # 確保尺寸是 8 的倍數，適配 U-Net 架構
                new_w = max(8, (new_w // 8) * 8)
                new_h = max(8, (new_h // 8) * 8)
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
        img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device) # (1, 3, H, W)
        
        # 進行指定次數的遞迴去模糊 (Multi-pass Deblurring)
        curr_tensor = img_tensor.clone()
        for it in range(args.iters):
            with torch.inference_mode():
                _, _, h, w = curr_tensor.shape
                if h <= args.tile_size or w <= args.tile_size:
                    with torch.amp.autocast('cuda', enabled=fp16_enabled):
                        curr_tensor = model(curr_tensor).float()
                else:
                    curr_tensor = tile_inference(model, curr_tensor, tile_size=args.tile_size, overlap=args.overlap, fp16=fp16_enabled)
                # 每次迭代後限制數值範圍在 0~1 之間，防止發散
                curr_tensor = torch.clamp(curr_tensor, 0, 1)
        
        pred = curr_tensor
        
        # 檢查並處理半精度下可能產生的 NaN 值 (防護機制)
        if torch.isnan(pred).any():
            print(f"\n⚠️ 警告：偵測到影像 {img_name} 的推理結果包含 NaN 值（可能由半精度 FP16 數值溢位引起）。")
            if fp16_enabled:
                print("💡 正在自動切換為單精度 (FP32) 重新進行推理以確保影像品質...")
                with torch.inference_mode():
                    if h <= args.tile_size or w <= args.tile_size:
                        pred = model(img_tensor).float()
                    else:
                        pred = tile_inference(model, img_tensor, tile_size=args.tile_size, overlap=args.overlap, fp16=False)
            else:
                print("❌ 錯誤：在單精度 (FP32) 下依然偵測到 NaN 值，請檢查權重或輸入圖像。")
        
        # 後處理並保存圖片
        pred_clip = torch.clamp(pred, 0, 1) + (0.5 / 255.0)
        pred_pil = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
        
        save_path = os.path.join(args.output_dir, img_name)
        pred_pil.save(save_path)
        
    print(f"處理完成！所有去模糊結果已儲存至: {args.output_dir}")

if __name__ == '__main__':
    main()
