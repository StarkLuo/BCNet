import os
from datetime import datetime
import cv2
import torch
from src import model
from torchvision import transforms
import argparse
import glob
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as transforms_true

class CustomDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms_true.Compose(transforms_) if transforms_ else None
        self.root = root
        # self.filelist = glob.glob(os.path.join(root, '*.jpg'))
        self.filelist = glob.glob(os.path.join(root, '*.jpg')) + glob.glob(os.path.join(root, '*.png'))
        self.filelist.sort()  #

    def __getitem__(self, index):
        img_path = self.filelist[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {'img': img, 'name': os.path.basename(img_path)}  

    def __len__(self):
        return len(self.filelist)


def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custompath', type=str, default='Coloring', help='Path to custom images for inference')
    parser.add_argument('--ckpt', type=str, default='./ckpts/BCNet-Tiny.pth', help='Path to the model checkpoint')
    parser.add_argument('--basic_c', type=int, default=16, help='')
    parser.add_argument('--range', type=int, default=1, help='')
    parser.add_argument('--device_num', type=str, default='cuda:0', help='cuda:0')
    parser.add_argument('--save_path', type=str, default='./results', help='Path to save the results')
    parser.add_argument('--invert', default=False,action='store_true', help='generate inverse edge map')
    return parser.parse_args()


def generator(args):
    device = args.device_num
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    trans = [transforms.ToTensor(), normalize]
    dataset = CustomDataset(args.custompath, trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    net = model.Contour_Detection(args).eval().to(device)
    state_dict = torch.load(args.ckpt)
    net.load_state_dict(state_dict)


    for i, data in enumerate(dataloader):
        img = data['img'].to(device)
        name = data['name'][0]
        # print(f"Processing {i+1}/{len(dataloader)}: {name}")
        with torch.no_grad():
            output = net(img).cpu().detach().numpy().squeeze()
        
        if args.invert:
            output = 1 - output

        save_path = os.path.join(args.save_path, name)
        cv2.imwrite(save_path, output*255)
        print(f"Processed {i+1}/{len(dataloader)}: {name} saved to {save_path}")


if __name__ == "__main__":
    args = args_init()
    generator(args)