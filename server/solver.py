from email.mime import image
from PIL import Image
import torch
from torchvision.utils import save_image
from model import Generator
from utils import Utils
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import attacks
from torchvision import transforms as T
from config import *


class Solver(Utils):

    def __init__(self):
        if use_cpu == True:
            # if use cpu force
            self.device = 'cpu'
        else:
            # use gpu
            self.device = 'cuda:' + \
                str(gpu_id) if torch.cuda.is_available() else 'cpu'

        self.G = Generator(64, 17, 6).to(self.device)

        # image regular
        self.regular_image_transform = []
        self.regular_image_transform.append(T.ToTensor())
        self.regular_image_transform.append(T.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        self.regular_image_transform = T.Compose(self.regular_image_transform)

        G_path = sorted(glob.glob(os.path.join(models_dir, '*G.ckpt')),
                        key=self.numericalSort)[0]

        if use_cpu == True:
            # use CPU
            self.G.load_state_dict(torch.load(G_path, map_location='cpu'))

            # Generator
            self.G = self.G.cpu()

        else:
            # use GPU
            self.G.load_state_dict(torch.load(
                G_path, map_location=f'cuda:{gpu_id}'))

            # Generator
            self.G = self.G.cuda(gpu_id)

        self.reference_expression_images = []

        with torch.no_grad():
            with open(attributes_path, 'r') as txt_file:
                csv_lines = txt_file.readlines()

                self.targets = torch.zeros(len(csv_lines), 17)
                self.input_images = torch.zeros(len(csv_lines), 3, 384, 384)

                for idx, line in enumerate(csv_lines):
                    splitted_lines = line.split(' ')
                    image_path = os.path.join(
                        attribute_images_dir, splitted_lines[0])
                    if use_cpu == True:
                        self.input_images[idx, :] = self.regular_image_transform(
                            Image.open(image_path)).cpu()
                    else:
                        self.input_images[idx, :] = self.regular_image_transform(
                            Image.open(image_path)).cuda()

                    self.reference_expression_images.append(splitted_lines[0])
                    self.targets[idx, :] = torch.Tensor(
                        np.array(list(map(lambda x: float(x)/5., splitted_lines[1::]))))

        # protect image
        self.pgd_attack = attacks.LinfPGDAttack(
            model=self.G, device=self.device)
        
    def textWatermark(self,img_src, dest, text, loc, fontsize=50, alpha=0.5):
        fig = plt.figure()
        # 读取图像
        plt.imshow(plt.imread(img_src))
        # 添加文字水印
        plt.text(loc[0], loc[1], text, fontsize=fontsize, alpha=alpha, color='gray')
        # 隐藏坐标轴
        plt.axis('off')
        # 保存图像
        plt.savefig(dest, dpi=fig.dpi, bbox_inches='tight')
        return fig

    def protect(self, args):

        protect_class_conditional = args['protect_class_conditional']
        protect_expression_seq = args['protect_expression_seq']
        image_name = args['image_name']


        image_path = protect_images_src_dir + '/' + image_name

        if use_cpu == True:
            image_to_animate = self.regular_image_transform(
                Image.open(image_path)).unsqueeze(0).cpu()
        else:
            image_to_animate = self.regular_image_transform(
                Image.open(image_path)).unsqueeze(0).cuda()



        if use_cpu == True:
            targets_au = self.targets[protect_expression_seq, :].unsqueeze(
                0).cpu()
        else:
            targets_au = self.targets[protect_expression_seq, :].unsqueeze(
                0).cuda()

        # gen-noAttack-image
        with torch.no_grad():
            # get attention_mask and color_regression
            resulting_images_att_noattack, resulting_images_reg_noattack = self.G(
                image_to_animate, targets_au)
            # get the ground-truth image
            if use_cpu == True:
                resulting_image_noattack = self.imFromAttReg(
                    resulting_images_att_noattack, resulting_images_reg_noattack, image_to_animate).cpu()
            else:
                resulting_image_noattack = self.imFromAttReg(
                    resulting_images_att_noattack, resulting_images_reg_noattack, image_to_animate).cuda()

        if protect_class_conditional == False:
            # Normal Attack
            x_adv, perturb = self.pgd_attack.perturb(
                image_to_animate, resulting_image_noattack, targets_au)
        elif protect_class_conditional == True:
            # Iterative Class Conditional
            if use_cpu == True:
                x_adv, perturb = self.pgd_attack.perturb_iter_class(
                    image_to_animate, image_to_animate, self.targets[:, :].cpu())
            else:
                x_adv, perturb = self.pgd_attack.perturb_iter_class(
                    image_to_animate, image_to_animate, self.targets[:, :].cuda())

        # Use this line if transferring attacks
        x_adv = image_to_animate + perturb

        # gen-Attack-image
        with torch.no_grad():
            resulting_images_att, resulting_images_reg = self.G(
                x_adv, targets_au)
            if use_cpu == True:
                resulting_image = self.imFromAttReg(
                    resulting_images_att, resulting_images_reg, x_adv).cpu()
            else:
                resulting_image = self.imFromAttReg(
                    resulting_images_att, resulting_images_reg, x_adv).cuda()

        protected_images_result_path = os.path.join(
            protected_images_result_dir, image_path.split('/')[-1])

        gen_protected_images_result_path = os.path.join(
            gen_protected_images_result_dir, image_path.split('/')[-1])

        gen_images_result_path = os.path.join(
            gen_images_result_dir, image_path.split('/')[-1])

        if protect_class_conditional == True:
            suffix = 'all.jpg'
        else:
            suffix = self.reference_expression_images[protect_expression_seq]
        # save attacked image
        save_image((x_adv+1)/2, protected_images_result_path.split('.jpg')
                [0] + '_' + suffix)

        # save result_image have been attacked
        save_image((resulting_image+1)/2, gen_protected_images_result_path.split(
            '.jpg')[0] + '_' + suffix)


        gen_face_path = gen_images_result_path.split('.jpg')[0] + '_' + suffix

        # save result_image that no attack
        save_image((resulting_image_noattack+1)/2, gen_face_path)
        
        # Add watermark
        #self.textWatermark(img_src=gen_face_path, dest=gen_face_path, text='FaceProtect'.encode('utf-8'), loc=[30, 50])
