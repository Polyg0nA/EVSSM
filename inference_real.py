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
    測試時自集成 (TTA) 推理。
    由於 EVSSM 模型內部硬編碼了 `assert b == 1`（在分塊裁切時限制了 Batch size 必須為 1），
    因此我們必須以序列化方式（一次處理一張幾何變化）來執行 8 次推理。
    我們在這裡加入內層 tqdm 進度條，讓您看清楚 TTA 融合的每一步進度。
    """
    outputs = []
    configs = []
    for flip in [False, True]:
        for rot in [0, 1, 2, 3]:
            configs.append((flip, rot))
            
    pbar_tta = tqdm(configs, desc="    TTA幾何融合", leave=False)
    for flip, rot in pbar_tta:
        x = img.clone()
        if rot > 0:
            x = torch.rot90(x, rot, [2, 3])
        if flip:
            x = torch.flip(x, [3])
        
        # 加上 torch.inference_mode() 避免保存計算圖（梯度），省下大量記憶體
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=fp16):
            pred = model(x).float()
        
        if flip:
            pred = torch.flip(pred, [3])
        if rot > 0:
            pred = torch.rot90(pred, -rot, [2, 3])
        outputs.append(pred)
        
    return torch.stack(outputs).mean(dim=0)

def deblur_one_step(model, img_tensor, args, device):
    """
    對輸入的 Tensor 執行單次去模糊推理（支援 TTA 與 Tiled Inference）。
    """
    _, _, h, w = img_tensor.shape
    use_tile = (h > args.tile_size or w > args.tile_size)
    
    with torch.inference_mode():  # 確保整個函數執行都在推理模式下，不保存梯度
        if args.tta:
            if use_tile:
                # 對分塊進行 TTA
                outputs_tta = []
                for flip in [False, True]:
                    for rot in [0, 1, 2, 3]:
                        x = img_tensor.clone()
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
                pred = torch.stack(outputs_tta).mean(dim=0)
            else:
                # 批次 TTA (平行加速)
                pred = tta_inference(model, img_tensor, fp16=False)
        else:
            if not use_tile:
                pred = model(img_tensor).float()
            else:
                pred = tile_inference(model, img_tensor, tile_size=args.tile_size, overlap=args.overlap, fp16=False)
            
    return torch.clamp(pred, 0, 1)

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
    parser.add_argument('--resize', type=str, default='-1', help='將影像長邊縮放到指定大小，支援多個尺寸以逗號分隔（如 "800,1200"），-1 表示不縮放。超大圖推薦縮放以獲得最佳效果與速度')
    parser.add_argument('--iters', type=int, default=1, help='遞迴去模糊次數。對於極大晃動可試試 2 或 3 次，但可能會使影像變平滑')
    parser.add_argument('--residual_mode', action='store_true', help='啟用低解析度引導殘差去模糊模式。這能讓大圖在輸出的同時，保有低解析度的強大去模糊效果與高解析度的細節')
    parser.add_argument('--tta', action='store_true', help='啟用測試時自集成 (Test-Time Augmentation, TTA) 幾何變換平均，可加強去模糊效果與減少偽影')
    parser.add_argument('--img_indices', type=str, default='', help='指定測試的圖片編號/索引（從 0 開始，用逗號分隔，例如 0,2,5）')
    parser.add_argument('--alpha', type=float, default=1.0, help='殘差融合放大係數。大於 1.0 (如 1.2 或 1.5) 可增強去模糊強度與銳利度')
    args = parser.parse_args()
    
    # 依使用者需求，統一使用 FP32 進行推理，不啟用 FP16
    fp16_enabled = False

    # 解析 --resize 參數，支援多個尺寸以逗號分隔
    scale_list = []
    if args.resize.strip():
        parts = args.resize.split(',')
        for p in parts:
            try:
                scale_val = int(p.strip())
                scale_list.append(scale_val)
            except ValueError:
                pass
    if not scale_list:
        scale_list = [-1]

    # 建立並清理輸出資料夾，避免殘留上一次執行的圖片
    os.makedirs(args.output_dir, exist_ok=True)
    for file in os.listdir(args.output_dir):
        file_path = os.path.join(args.output_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"⚠️ 警告：無法刪除舊檔案 {file_path}: {e}")

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

    for img_path in img_paths:  # 主進度條移到手動或外層，因內層有多尺度進度條
        img_name = os.path.basename(img_path)
        print(f"\n🚀 正在處理影像: {img_name}")
        
        # 讀取圖片並設定解析度處理
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        
        # 原始高解析度影像 Tensor
        img_tensor_high = F.to_tensor(img_pil).unsqueeze(0).to(device)
        
        # 定義多尺度處理閉包
        def process_image_scales():
            # 檢查是否需要啟用殘差模式：
            # 1. 使用者明確啟用 --residual_mode
            # 2. 或者有多個尺度需要融合
            use_residual = args.residual_mode or len(scale_list) > 1
            
            # 如果不需要殘差模式（即單一尺度且未使用殘差模式）
            if not use_residual:
                scale_val = scale_list[0]
                if scale_val > 0 and max(orig_w, orig_h) > scale_val:
                    scale = scale_val / max(orig_w, orig_h)
                    new_w = max(8, ((int(orig_w * scale)) // 8) * 8)
                    new_h = max(8, ((int(orig_h * scale)) // 8) * 8)
                    img_pil_low = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    curr_tensor = F.to_tensor(img_pil_low).unsqueeze(0).to(device)
                    scale_desc = f"{new_w}x{new_h}"
                else:
                    curr_tensor = img_tensor_high.clone()
                    scale_desc = "原圖"
                
                pbar_it = tqdm(range(args.iters), desc=f"  -> {img_name} ({scale_desc}) 疊代", leave=False)
                for it in pbar_it:
                    pbar_it.set_postfix(step=f"{it+1}/{args.iters}")
                    curr_tensor = deblur_one_step(model, curr_tensor, args, device)
                return curr_tensor
            
            # 交織多尺度殘差疊代反饋機制
            curr_high_tensor = img_tensor_high.clone()
            pbar_it = tqdm(range(args.iters), desc=f"  -> {img_name} (多尺度交織疊代)", leave=False)
            for it in pbar_it:
                pbar_it.set_postfix(step=f"{it+1}/{args.iters}")
                upsampled_residuals = []
                
                for scale_val in scale_list:
                    # 1. 下採樣當前的中間高解析度 Tensor 到該尺度
                    if scale_val > 0 and max(orig_w, orig_h) > scale_val:
                        scale = scale_val / max(orig_w, orig_h)
                        new_w = max(8, ((int(orig_w * scale)) // 8) * 8)
                        new_h = max(8, ((int(orig_h * scale)) // 8) * 8)
                        img_tensor_low = torch.nn.functional.interpolate(
                            curr_high_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
                        )
                    else:
                        img_tensor_low = curr_high_tensor.clone()
                    
                    # 2. 進行單次模型去模糊推理
                    pred_low = deblur_one_step(model, img_tensor_low, args, device)
                    
                    # 3. 計算本次尺度的去模糊殘差 (去模糊結果 - 本次輸入)
                    residual_low = pred_low - img_tensor_low
                    
                    # 4. 上採樣殘差回原始尺寸
                    if residual_low.shape[2] != orig_h or residual_low.shape[3] != orig_w:
                        residual_high = torch.nn.functional.interpolate(
                            residual_low, size=(orig_h, orig_w), mode='bicubic', align_corners=False
                        )
                    else:
                        residual_high = residual_low
                    
                    upsampled_residuals.append(residual_high)
                
                # 融合本輪所有尺度的殘差 (取平均)
                fused_residual = torch.stack(upsampled_residuals).mean(dim=0)
                
                # 將融合殘差加回當前大圖，更新 curr_high_tensor 做為下一輪疊代的輸入
                curr_high_tensor = torch.clamp(curr_high_tensor + args.alpha * fused_residual, 0, 1)
                
                # 釋放 GPU 顯存快取，避免多尺度迭代累積顯存
                torch.cuda.empty_cache()
                
            return curr_high_tensor
        
        # 執行主處理流程
        pred = process_image_scales()
        
        # 檢查並處理可能產生的 NaN 值 (防護機制)
        if torch.isnan(pred).any():
            print(f"❌ 錯誤：偵測到影像 {img_name} 的推理結果包含 NaN 值，請檢查權重或輸入圖像。")
        
        # 後處理並保存圖片
        pred_clip = torch.clamp(pred, 0, 1) + (0.5 / 255.0)
        pred_pil = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
        
        save_path = os.path.join(args.output_dir, img_name)
        pred_pil.save(save_path)
        
    print(f"處理完成！所有去模糊結果已儲存至: {args.output_dir}")

if __name__ == '__main__':
    main()
