import lpips
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import csv
import argparse

def load_image(path, to_rgb=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def compute_lpips(loss_fn, img1, img2, device):
    t1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().to(device)
    t2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float().to(device)
    t1 = (t1 / 127.5 - 1.0)
    t2 = (t2 / 127.5 - 1.0)
    with torch.no_grad():
        return loss_fn(t1, t2).item()

def safe_ssim(img1, img2, data_range=255):
    h, w = img1.shape[:2]
    default_ws = 7
    max_ws = min(h, w)
    if max_ws < default_ws:
        ws = max_ws if max_ws % 2 == 1 else max_ws - 1
        ws = max(ws, 3)
    else:
        ws = default_ws

    return compare_ssim(
        img1, img2,
        data_range=data_range,
        channel_axis=2,
        win_size=ws
    )

def evaluate_all_methods(gt_dir="gt", method_dirs=["bcnet", "bcnets", "bcnett", "PEdger", "pidinet"]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='vgg').to(device)
    
    # Get all ground truth images
    gt_images = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    results = {}
    print(f"\nEvaluating {len(method_dirs)} methods...")
    
    # Method-level progress bar
    for method in tqdm(method_dirs, desc="Method Progress"):
        if not os.path.exists(method):
            continue
            
        method_results = []
        gt_images_pbar = tqdm(gt_images, desc=f"Processing {method}", leave=False)
        
        for gt_img in gt_images_pbar:
            img_name = os.path.splitext(gt_img)[0]
            orig_path = os.path.join(gt_dir, gt_img)
            sub_dir = os.path.join(method, img_name)
            
            if not os.path.isdir(sub_dir):
                continue
                
            # Evaluate this subfolder
            metrics = evaluate_folder(orig_path, sub_dir, save_csv=False, loss_fn=loss_fn_alex, device=device)
            method_results.append(metrics)
            gt_images_pbar.set_postfix_str(f"Current: {img_name}")
        
        # Calculate mean and std for this method
        if method_results:
            arr = np.array(method_results)
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            results[method] = (means, stds, method_results)  # Save mean, std and raw data
    
    # Print all method results (with mean and std)
    print("\n=== Method Comparison Results ===")
    print(f"{'Method':<10} {'PSNR(mean±std)':<20} {'SSIM(mean±std)':<20} {'LPIPS(mean±std)':<20}")
    print("-"*70)
    for method, (means, stds, _) in results.items():
        psnr_mean, ssim_mean, lpips_mean = means
        psnr_std, ssim_std, lpips_std = stds
        print(f"{method:<10} {f'{psnr_mean:.2f}±{psnr_std:.2f}':<20} {f'{ssim_mean:.4f}±{ssim_std:.4f}':<20} {f'{lpips_mean:.4f}±{lpips_std:.4f}':<20}")

def evaluate_folder(orig_path, new_dir, save_csv=False, csv_path="metrics.csv",
                   loss_fn=None, device=None):
    """Modified evaluation function that supports external LPIPS calculator"""
    if loss_fn is None or device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = lpips.LPIPS(net='vgg').to(device)

    orig = load_image(orig_path)
    oh, ow = orig.shape[:2]
    results = []

    for fname in sorted(os.listdir(new_dir)):
        fpath = os.path.join(new_dir, fname)
        try:
            img = load_image(fpath)
        except FileNotFoundError:
            continue

        if img.shape[:2] != (oh, ow):
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        psnr = compare_psnr(orig, img, data_range=255)
        ssim = safe_ssim(orig, img, data_range=255)
        lp   = compute_lpips(loss_fn, orig, img, device)

        results.append((fname, psnr, ssim, lp))

    # Convert to NumPy array for statistics
    arr = np.array([[r[1], r[2], r[3]] for r in results])
    means = arr.mean(axis=0) if len(arr) > 0 else np.zeros(3)
    
    if save_csv:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'psnr', 'ssim', 'lpips'])
            writer.writerows(results)
        print(f"\nAll results saved to `{csv_path}`")
    
    return means

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate image quality metrics for different methods')
    parser.add_argument('--methods', nargs='+', default=["bcnet",],
                       help='List of method directories to evaluate')
    parser.add_argument('--gt_dir', default="gt", help='Ground truth images directory')
    args = parser.parse_args()
    
    evaluate_all_methods(gt_dir=args.gt_dir, method_dirs=args.methods)