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

def tta_inference(model, img, fp16=False):
    """
    批次測試時自集成 (Batch TTA) 推理。
    將 8 種幾何變換組合打包成一個 Batch (Batch size = 8) 一次送入 GPU，
    以充分利用剩餘 VRAM (讓 VRAM 吃到 10GB~12GB)，大幅提升並行計算速度！
    """
    inputs = []
    configs = []  # 儲存 (flip, rot) 的幾何配置
    for flip in [False, True]:
        for rot in [0, 1, 2, 3]:
            x = img.clone()
            if rot > 0:
                x = torch.rot90(x, rot, [2, 3])
            if flip:
                x = torch.flip(x, [3])
            inputs.append(x)
            configs.append((flip, rot))
            
    # 將 8 張圖像在 batch 維度拼接，shape: (8, C, H, W)
    batch_x = torch.cat(inputs, dim=0)
    
    # 批次送入 GPU 運算
    with torch.amp.autocast('cuda', enabled=fp16):
        batch_pred = model(batch_x).float() # shape: (8, C, H, W)
        
    # 還原每張圖的幾何變換
    outputs = []
    for i in range(8):
        flip, rot = configs[i]
        pred = batch_pred[i:i+1] # 取出單張 tensor, shape: (1, C, H, W)
        if flip:
            pred = torch.flip(pred, [3])
        if rot > 0:
            pred = torch.rot90(pred, -rot, [2, 3])
        outputs.append(pred)
        
    return torch.stack(outputs).mean(dim=0)

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
    parser.add_argument('--residual_mode', action='store_true', help='啟用低解析度引導殘差去模糊模式。這能讓大圖在輸出的同時，保有低解析度的強大去模糊效果與高解析度的細節')
    parser.add_argument('--tta', action='store_true', help='啟用測試時自集成 (Test-Time Augmentation, TTA) 幾何變換平均，可加強去模糊效果與減少偽影')
    parser.add_argument('--img_indices', type=str, default='', help='指定測試的圖片編號/索引（從 0 開始，用逗號分隔，例如 0,2,5）')
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
    
    # 先進行檔案排序，確保編號索引順序固定
    img_paths = sorted(img_paths)
    
    # 優先根據指定圖片編號進行篩選
    if args.img_indices:
        try:
            indices = [int(x.strip()) for x in args.img_indices.split(',')]
            valid_paths = []
            for idx in indices:
                if 0 <= idx < len(img_paths):
                    valid_paths.append(img_paths[idx])
                else:
                    print(f"⚠️ 警告：索引 {idx} 超出範圍（共有 {len(img_paths)} 張影像，有效索引為 0 到 {len(img_paths)-1}）")
            img_paths = valid_paths
            print(f"依設定只處理索引為 {indices} 的影像，開始去模糊推理...")
        except ValueError:
            print("⚠️ 警告：--img_indices 格式錯誤，請使用逗號分隔的整數（例如 0,2,5）！將不限制圖片編號。")
            
    # 若無指定編號，才限制處理前 N 張圖片
    elif args.num_images > 0:
        img_paths = img_paths[:args.num_images]
        print(f"依設定只處理前 {args.num_images} 張影像，開始去模糊推理...")
    else:
        print(f"找到 {len(img_paths)} 張影像，開始去模糊推理...")

    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)
        
        # 讀取圖片並設定解析度處理
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        
        # 原始高解析度影像 Tensor
        img_tensor_high = F.to_tensor(img_pil).unsqueeze(0).to(device)
        
        # 進行縮放 (適配感受野)
        if args.resize > 0 and max(orig_w, orig_h) > args.resize:
            scale = args.resize / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            # 確保尺寸是 8 的倍數
            new_w = max(8, (new_w // 8) * 8)
            new_h = max(8, (new_h // 8) * 8)
            img_pil_low = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img_tensor_low = F.to_tensor(img_pil_low).unsqueeze(0).to(device)
        else:
            img_tensor_low = img_tensor_high.clone()
            
        # 進行指定次數的遞迴去模糊 (針對低解析度影像進行，其感受野最契合)
        curr_tensor = img_tensor_low.clone()
        pbar_it = tqdm(range(args.iters), desc=f"  -> {img_name} 疊代", leave=False)
        for it in pbar_it:
            pbar_it.set_postfix(step=f"{it+1}/{args.iters}")
            with torch.inference_mode():
                _, _, h, w = curr_tensor.shape
                use_tile = (h > args.tile_size or w > args.tile_size)
                
                if args.tta:
                    if use_tile:
                        # 對分塊進行 TTA
                        outputs_tta = []
                        for flip in [False, True]:
                            for rot in [0, 1, 2, 3]:
                                x = curr_tensor.clone()
                                if rot > 0:
                                    x = torch.rot90(x, rot, [2, 3])
                                if flip:
                                    x = torch.flip(x, [3])
                                pred_t = tile_inference(model, x, tile_size=args.tile_size, overlap=args.overlap, fp16=fp16_enabled)
                                if flip:
                                    pred_t = torch.flip(pred_t, [3])
                                if rot > 0:
                                    pred_t = torch.rot90(pred_t, -rot, [2, 3])
                                outputs_tta.append(pred_t)
                        curr_tensor = torch.stack(outputs_tta).mean(dim=0)
                    else:
                        # 批次 TTA (平行加速)
                        curr_tensor = tta_inference(model, curr_tensor, fp16=fp16_enabled)
                else:
                    if not use_tile:
                        with torch.amp.autocast('cuda', enabled=fp16_enabled):
                            curr_tensor = model(curr_tensor).float()
                    else:
                        curr_tensor = tile_inference(model, curr_tensor, tile_size=args.tile_size, overlap=args.overlap, fp16=fp16_enabled)
                # 每次迭代後限制數值範圍在 0~1 之間，防止發散
                curr_tensor = torch.clamp(curr_tensor, 0, 1)
        
        pred_low = curr_tensor
        
        # 殘差模式融合 (將低解析度下獲得的去模糊 Delta 上採樣並加回原大圖)
        if args.residual_mode and args.resize > 0 and max(orig_w, orig_h) > args.resize:
            residual_low = pred_low - img_tensor_low
            # 上採樣殘差到原始尺寸
            residual_high = torch.nn.functional.interpolate(
                residual_low, size=(orig_h, orig_w), mode='bilinear', align_corners=False
            )
            pred = img_tensor_high + residual_high
            pred = torch.clamp(pred, 0, 1)
        else:
            pred = pred_low
        
        # 檢查並處理半精度下可能產生的 NaN 值 (防護機制)
        if torch.isnan(pred).any():
            print(f"\n⚠️ 警告：偵測到影像 {img_name} 的推理結果包含 NaN 值（可能由半精度 FP16 數值溢位引起）。")
            if fp16_enabled:
                print("💡 正在自動切換為單精度 (FP32) 重新進行推理以確保影像品質...")
                with torch.inference_mode():
                    curr_tensor_fp32 = img_tensor_low.clone()
                    pbar_it_fb = tqdm(range(args.iters), desc=f"  -> [修復] {img_name} 疊代", leave=False)
                    for it in pbar_it_fb:
                        pbar_it_fb.set_postfix(step=f"{it+1}/{args.iters}")
                        _, _, h, w = curr_tensor_fp32.shape
                        use_tile = (h > args.tile_size or w > args.tile_size)
                        
                        if args.tta:
                            if use_tile:
                                outputs_tta = []
                                for flip in [False, True]:
                                    for rot in [0, 1, 2, 3]:
                                        x = curr_tensor_fp32.clone()
                                        if rot > 0:
                                            x = torch.rot90(x, rot, [2, 3])
                                        if flip:
                                            x = torch.flip(x, [3])
                                        pred_t = tile_inference(model, x, tile_size=args.tile_size, overlap=args.overlap, fp16=False)
                                        if flip:
                                            pred_t = torch.flip(pred_t, [3])
                                        if rot > 0:
                                            pred_t = torch.rot90(pred_t, -rot, [2, 3])
                                        outputs_tta.append(pred_t)
                                curr_tensor_fp32 = torch.stack(outputs_tta).mean(dim=0)
                            else:
                                curr_tensor_fp32 = tta_inference(model, curr_tensor_fp32, fp16=False)
                        else:
                            if not use_tile:
                                curr_tensor_fp32 = model(curr_tensor_fp32).float()
                            else:
                                curr_tensor_fp32 = tile_inference(model, curr_tensor_fp32, tile_size=args.tile_size, overlap=args.overlap, fp16=False)
                        curr_tensor_fp32 = torch.clamp(curr_tensor_fp32, 0, 1)
                    
                    if args.residual_mode and args.resize > 0 and max(orig_w, orig_h) > args.resize:
                        residual_low = curr_tensor_fp32 - img_tensor_low
                        residual_high = torch.nn.functional.interpolate(
                            residual_low, size=(orig_h, orig_w), mode='bilinear', align_corners=False
                        )
                        pred = img_tensor_high + residual_high
                        pred = torch.clamp(pred, 0, 1)
                    else:
                        pred = curr_tensor_fp32
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
