"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, Memory
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
from deeplab import Deeplab #lys
from torch.autograd import Variable

def semantic(init_weights=None):
        init_weights = './cyclegan_sem_model.pth'
        model = Deeplab(num_classes=19)
        if init_weights is not None:
            saved_state_dict = torch.load(init_weights, map_location=lambda storage, loc: storage)
            model.load_state_dict(saved_state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.cuda()
        # model = torch.nn.DataParallel(model, gpu_ids)
        return model

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        #self.mem_a = Memory()
        #self.mem_b = Memory()
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.instancenorm = nn.InstanceNorm2d(19)

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            # pdb.set_trace()
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.semantic = semantic()
    
    def compute_semantic_loss(self, img_feat, target_feat):
        # pdb.set_trace()
        # print('')
        return torch.mean((self.instancenorm(img_feat) - self.instancenorm(target_feat)) ** 2)

    def compute_semantic_feat(self, img):
        img_vgg = self.img_preprocess(img)
        img_feat = self.semantic(img_vgg)
        return img_feat  

    def img_preprocess(self, batch):
        tensortype = type(batch.data)
        (r, g, b) = torch.chunk(batch, 3, dim=1)
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
        batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 104.00698793
        mean[:, 1, :, :] = 116.66876762
        mean[:, 2, :, :] = 122.67891434
        batch = batch.sub(Variable(mean).to(batch.device))  # subtract mean
        return batch

    def supervised_contrastive_criterion(self, q_f, q_l, k_f, k_l):
        """
        q_f: query feature(memory style feature), k_f: key feature(image style feature, detached)
        q_l: query semantic label, k_l: key semantic label
        """
        n, c, h, w = q_f.size()
        q_f = q_f.view((n, c, h * w)).transpose(2, 1) # (N, HW, C)
        k_f = k_f.view((n, c, h * w)) # (N, C, HW)
        q_l = F.interpolate(q_l.unsqueeze(1).float(), (h, w), mode='nearest').long()
        k_l = F.interpolate(k_l.unsqueeze(1).float(), (h, w), mode='nearest').long()
        q_l = q_l.view((n, 1, h * w)).transpose(2, 1) # (N, HW, 1) 
        k_l = k_l.view((n, 1, h * w)) # (N, 1, HW)

        mask = torch.eq(q_l, k_l) # (N, HW, HW)
        global_logits = torch.bmm(q_f, k_f) # (N, HW, HW)
        global_logits = global_logits.clamp(-1, 1) / 0.07
        exp_logits = torch.exp(global_logits)
        #exp_logits = torch.clamp(torch.exp(global_logits / 0.07), min=1e-5)

        log_prob = global_logits - torch.log((~mask * exp_logits).sum(2, keepdim=True) + exp_logits)
        mean_log_prob_pos = - (mask * log_prob).sum(2) / mask.sum(2)
        
        total = mask.sum(2)
        valid_mask = (total > 0)
        mean_log_prob_pos = mean_log_prob_pos * valid_mask
        loss = torch.sum(mean_log_prob_pos) / (h * w - (~valid_mask).sum())
        # print(loss)

        return loss

    def contrastive_criterion(self, q_f, k_f, norm=False):
        """
        q_f: query feature, k_f: key_feature
        """
        if norm:
            q_f, k_f = F.normalize(q_f, dim=1), F.normalize(k_f, dim=1)
        n, c, h, w = q_f.size()
        q_f = q_f.view((n, c, h * w)).transpose(2, 1) # (N, HW, C)
        k_f = k_f.view((n, c, h * w)) # (N, C, HW)
        #pdb.set_trace()
        global_logits = torch.bmm(q_f, k_f)
        global_logits = global_logits.clamp(-1, 1) / 0.07
        exp_logits = torch.exp(global_logits)

        mask = torch.eye(exp_logits.size()[1], dtype=torch.bool, device=q_f.device).unsqueeze(0)
        log_prob = global_logits - torch.log((~mask * exp_logits).sum(2, keepdim=True) + exp_logits)
        mean_log_prob_pos = -(mask * log_prob).sum(2) / mask.sum(2)
        loss = torch.mean(mean_log_prob_pos)

        return loss

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, data, hyperparameters):
        # pdb.set_trace()
        x_a, l_a, ol_a = data['D_image'].cuda().detach(), data['D_label'].cuda().detach(), data['D_onehot_label'].cuda().detach()
        x_b, l_b, ol_b = data['N_image'].cuda().detach(), data['N_label'].cuda().detach(), data['N_onehot_label'].cuda().detach()

        self.gen_opt.zero_grad()
        
        # encode
        c_a, s_a = self.gen_a.encode(x_a, ol_a) # A domain content, A domain style
        c_b, s_b = self.gen_b.encode(x_b, ol_b) # B domain content, B domain style

        # memory read
        ms_b = self.gen_a.read(c_a, l_a) # B domain memory style
        ms_a = self.gen_b.read(c_b, l_b) # A domain memory style

        # style normalization
        s_a, s_b = F.normalize(s_a, dim=1), F.normalize(s_b, dim=1)
        ms_a, ms_b = F.normalize(ms_a, dim=1), F.normalize(ms_b, dim=1)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, None, s_a, l_a) # A domain generation(content, B style or None, A style, label)
        x_b_recon = self.gen_b.decode(c_b, None, s_b, l_b) # B domain generation(content, A style or None, B style, label)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_b, ms_a, l_b) # A domain generation(content, B style or None, A style, label)
        x_ab = self.gen_b.decode(c_a, s_a, ms_b, l_a) # B domain gneeration(content, A style or None, B style, label)

        # encode again
        c_b_recon, _ = self.gen_a.encode(x_ba, ol_b)
        c_a_recon, _ = self.gen_b.encode(x_ab, ol_a)

        # memory read
        ms_a_recon = self.gen_b.read(c_a_recon, l_a) # A domain memory style
        ms_b_recon = self.gen_a.read(c_b_recon, l_b) # B domain memory style 

        # style normalization
        ms_a_recon, ms_b_recon = F.normalize(ms_a_recon, dim=1), F.normalize(ms_b_recon, dim=1)

        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, None, ms_a_recon, l_a) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, None, ms_b_recon, l_b) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a) # ok
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b) # ok
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0 # ok
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0 # ok

        # semantic loss
        real_A_feat, rec_A_feat = self.compute_semantic_feat(x_a), self.compute_semantic_feat(x_aba)
        real_B_feat, rec_B_feat = self.compute_semantic_feat(x_b), self.compute_semantic_feat(x_bab)
        fake_A_feat, fake_B_feat = self.compute_semantic_feat(x_ba), self.compute_semantic_feat(x_ab)
        
        self.loss_rec_sem_A = self.compute_semantic_loss(rec_A_feat, real_A_feat) #* lambda_A * self.opt.lambda_semantic #lys
        self.loss_rec_sem_B = self.compute_semantic_loss(rec_B_feat, real_B_feat) #* lambda_B * self.opt.lambda_semantic #lys
        
        self.loss_sem_A = self.compute_semantic_loss(fake_B_feat, real_A_feat) #* 0.1 #lys
        self.loss_sem_B = self.compute_semantic_loss(fake_A_feat, real_B_feat) #* 0.1 #lys

        # contrastive loss
        self.loss_gen_recon_s_a = self.contrastive_criterion(ms_a_recon, s_a.detach()) # ok
        self.loss_gen_recon_s_b = self.contrastive_criterion(ms_b_recon, s_b.detach()) # ok
        self.loss_gen_recon_c_a = self.contrastive_criterion(c_a_recon, c_a.detach(), True) # ok
        self.loss_gen_recon_c_b = self.contrastive_criterion(c_b_recon, c_b.detach(), True) # ok

        # pdb.set_trace()
       
        # if needed for print

        # print('style recon loss a b: {} {}'.format(self.loss_gen_recon_s_a.data, self.loss_gen_recon_s_b.data))
        # print('content recon loss a b: {} {}'.format(self.loss_gen_recon_c_a.data, self.loss_gen_recon_c_b.data))
        # print('semantic loss a b: {} {}'.format(self.loss_sem_A.data, self.loss_sem_A.data))
        # print('semantic recon loss a b: {} {}'.format(self.loss_rec_sem_A.data, self.loss_rec_sem_B.data))

        # supervised contrastive loss
        #self.loss_gen_contrastive_loss_s_a = self.supervised_contrastive_criterion(ms_a, l_b, s_b.detach(), l_b)
        #self.loss_gen_contrastive_loss_s_b = self.supervised_contrastive_criterion(ms_b, l_a, s_a.detach(), l_a)

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        
        # pdb.set_trace()

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['sem_a_w'] * self.loss_sem_A + \
                              hyperparameters['sem_b_w'] * self.loss_sem_B + \
                              hyperparameters['sem_recon_a_w'] * self.loss_rec_sem_A + \
                              hyperparameters['sem_recon_b_w'] * self.loss_rec_sem_B
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, data):
        x_a, l_a, ol_a = data['D_image'].cuda().detach(), data['D_label'].cuda().detach(), data['D_onehot_label'].cuda().detach()
        x_b, l_b, ol_b = data['N_image'].cuda().detach(), data['N_label'].cuda().detach(), data['N_onehot_label'].cuda().detach()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        
        self.eval()
    
        for i in range(x_a.size(0)):
            c_a, s_a = self.gen_a.encode(x_a[i].unsqueeze(0), ol_a[i].unsqueeze(0))
            c_b, s_b = self.gen_b.encode(x_b[i].unsqueeze(0), ol_b[i].unsqueeze(0))
            
            ms_b = self.gen_a.read(c_a, l_a)
            ms_a = self.gen_b.read(c_b, l_b)

            s_a, s_b = F.normalize(s_a, dim=1), F.normalize(s_b, dim=1)
            ms_a, ms_b = F.normalize(ms_a, dim=1), F.normalize(ms_b, dim=1)

            x_a_recon.append(self.gen_a.decode(c_a, None, s_a, l_a))
            x_b_recon.append(self.gen_b.decode(c_b, None, s_b, l_b))

            x_ba.append(self.gen_a.decode(c_b, s_b, ms_a, l_b))
            x_ab.append(self.gen_b.decode(c_a, s_a, ms_b, l_a))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)

        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, data, hyperparameters):
        # pdb.set_trace()
        x_a, l_a, ol_a = data['D_image'].cuda().detach(), data['D_label'].cuda().detach(), data['D_onehot_label'].cuda().detach()
        x_b, l_b, ol_b = data['N_image'].cuda().detach(), data['N_label'].cuda().detach(), data['N_onehot_label'].cuda().detach()

        self.dis_opt.zero_grad()
        #s_a = Variable(torch.randn(i_a.size(0), self.style_dim, 1, 1).cuda())
        #s_b = Variable(torch.randn(i_b.size(0), self.style_dim, 1, 1).cuda())
        
        # encode
        c_a, s_a = self.gen_a.encode(x_a, ol_a)
        c_b, s_b = self.gen_b.encode(x_b, ol_b)
        
        # memory read
        ms_b = self.gen_a.read(c_a, l_a)
        ms_a = self.gen_b.read(c_b, l_b)

        # style normalization
        s_a, s_b = F.normalize(s_a, dim=1), F.normalize(s_b, dim=1)
        ms_a, ms_b = F.normalize(ms_a, dim=1), F.normalize(ms_b, dim=1)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_b, ms_a, l_b)
        x_ab = self.gen_b.decode(c_a, s_a, ms_b, l_a)
        
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
