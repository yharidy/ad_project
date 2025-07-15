import argparse
import os
import mmcv
from tqdm import tqdm
from mmengine.config import Config
from mmdet3d.apis import init_model, inference_detector

def run_inference(input_dir, output_dir, config_path, checkpoint_path, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    # Load BevFormer model
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.mdata.test.test_mode = True
    model = init_model(cfg, checkpoint_path, device=device)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ])
    for img_name in tqdm(image_files, desc="Running camera perception"):
        img_path = os.path.join(input_dir, img_name)
        result, data = inference_detector(model, img_path)
        
        # Save results
        output_file = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_result.pkl")
        model.show_results(data, result, out_file=output_file, show=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required= True, help="Path to input images")
    parser.add_argument('--output_dir', type=str, default='outputs', help='Where to save results')
    parser.add_argument('--cfg', type=str, default='camera_perception/configs/bevformer_r50.py')
    parser.add_argument('--ckpt', type=str, default='camera_perception/checkpoints/bevformer_r50_epoch_24.pth')
    args = parser.parse_args()