
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn



class CMBFSCNN_level3(nn.Module):
    def __init__(self, in_channels = 10, out_channels=1, n_feats=16):
        super(CMBFSCNN_level3, self).__init__()
        self.adnet = BRDNet(n_feats=n_feats, in_channel = in_channels*2)
        self.unet_enconder_patch1 = Encoder(in_channels = in_channels)
        self.unet_deconder_patch1 = Decoder(n_feat = 256, out_channels = in_channels)
        self.unet_enconder_patch2 = Encoder(in_channels = 2 *in_channels)
        self.unet_deconder_patch2 = Decoder(n_feat=256*2, out_channels = in_channels)

        self.unet_enconder = Encoder(in_channels=2 * in_channels)
        self.unet_deconder = Decoder(n_feat=256 * 3, out_channels = out_channels)
        self.cov = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, padding=1, stride=1))
        self.over_pix = 64

    def forward(self, x):
        H = x.size(2)
        W = x.size(3)

        #  Patches for Stage 3
        x3l_img = x[:, :, :, 0:int(W / 2)+self.over_pix]
        x3r_img = x[:, :, :, int(W / 2)-self.over_pix:W]

        #  Patches for Stage 2
        x2ltop_img = x[:, :, 0:int(H / 2)+self.over_pix, 0:int(W / 2)+self.over_pix]
        x2rtop_img = x[:, :, 0:int(H / 2)+self.over_pix, int(W / 2)-self.over_pix:W]
        x2lbot_img = x[:, :, int(H / 2)-self.over_pix:H, 0:int(W / 2)+self.over_pix]
        x2rbot_img = x[:, :, int(H / 2)-self.over_pix:H, int(W / 2)-self.over_pix:W]


        # stage2 encoder
        feat2_enc_ltop = self.unet_enconder_patch1(x2ltop_img)
        feat2_enc_rtop = self.unet_enconder_patch1(x2rtop_img)
        feat2_enc_lbot = self.unet_enconder_patch1(x2lbot_img)
        feat2_enc_rbot = self.unet_enconder_patch1(x2rbot_img)
        del x2ltop_img
        del x2rtop_img
        del x2lbot_img
        del x2rbot_img

        feat2_enc_l = [torch.cat((k, v), 2) for k, v in zip(feat2_enc_ltop, feat2_enc_lbot)]
        feat2_enc_r = [torch.cat((k, v), 2) for k, v in zip(feat2_enc_rtop, feat2_enc_rbot)]
        feat2_skip_l = feat2_enc_l[-1]

        W1 = int(W / 4 / 2 ** 5)
        over_pix_1 = int(self.over_pix / 2 ** 5)
        H3 = int(H/2/2**5)
        feat2_skip_l_1 = feat2_skip_l[:,:,0:H3,:]
        feat2_skip_l_2 = feat2_skip_l[:, :, H3+over_pix_1 * 2:, :]
        feat2_skip_l_1[:,:,H3-over_pix_1:,:] = (feat2_skip_l_1[:,:,H3-over_pix_1:,:] + feat2_skip_l[:,:,H3 + over_pix_1:H3 + over_pix_1 * 2,:])/2
        feat2_skip_l_2[:,:,0:over_pix_1,:] = (feat2_skip_l_2[:,:,0:over_pix_1,:] + feat2_skip_l[:,:,H3:H3 + over_pix_1,:])/2
        feat2_skip_l = torch.cat((feat2_skip_l_1, feat2_skip_l_2),2)
        # feat2_enc_l[4] = feat2_skip_l
        del feat2_skip_l_1
        del feat2_skip_l_2




        feat2_skip_r = feat2_enc_r[-1]
        feat2_skip_r_1 = feat2_skip_r[:, :, 0:H3, :]
        feat2_skip_r_2 = feat2_skip_r[:, :, H3+over_pix_1 * 2:, :]
        feat2_skip_r_1[:, :, H3-over_pix_1:, :] = (feat2_skip_r_1[:, :, H3-over_pix_1:, :] + feat2_skip_r[:, :, H3 + over_pix_1:H3 + over_pix_1 * 2, :]) / 2
        feat2_skip_r_2[:, :, 0:over_pix_1, :] = (feat2_skip_r_2[:, :, 0:over_pix_1, :] + feat2_skip_r[:, :, H3:H3 + over_pix_1, :]) / 2
        feat2_skip_r = torch.cat((feat2_skip_r_1, feat2_skip_r_2), 2)
        # feat2_enc_r[4] = feat2_skip_r
        del feat2_skip_r_1
        del feat2_skip_r_2

        # stage2 decoder
        feat2_dec_l = self.unet_deconder_patch1(feat2_enc_l)
        feat2_dec_r = self.unet_deconder_patch1(feat2_enc_r)
        del feat2_enc_l
        del feat2_enc_r
        H4 = int(H/2)
        feat2_dec_l_1 = feat2_dec_l[:, :, 0:H4, :]
        feat2_dec_l_2 = feat2_dec_l[:, :, H4 + self.over_pix*2:, :]
        feat2_dec_l_1[:, :, H4 - self.over_pix:, :] = (feat2_dec_l_1[:, :, H4 - self.over_pix:, :] + feat2_dec_l[:, :, H4 + self.over_pix:H4 + self.over_pix*2,
                                                                                 :]) / 2
        feat2_dec_l_2[:, :, 0:self.over_pix, :] = (feat2_dec_l_1[:, :, 0:self.over_pix, :] + feat2_dec_l[:, :, H4:H4 + self.over_pix, :]) / 2
        feat2_dec_l = torch.cat((feat2_dec_l_1, feat2_dec_l_2), 2)
        del feat2_dec_l_1
        del feat2_dec_l_2


        feat2_dec_r_1 = feat2_dec_r[:, :, 0:H4, :]
        feat2_dec_r_2 = feat2_dec_r[:, :, H4 + self.over_pix*2:, :]
        feat2_dec_r_1[:, :, H4 - self.over_pix:, :] = (feat2_dec_r_1[:, :, H4 - self.over_pix:, :] + feat2_dec_r[:, :, H4 + self.over_pix:H4 + self.over_pix*2,
                                                                                 :]) / 2
        feat2_dec_r_2[:, :, 0:self.over_pix, :] = (feat2_dec_r_1[:, :, 0:self.over_pix, :] + feat2_dec_r[:, :, H4:H4 + self.over_pix, :]) / 2
        feat2_dec_r = torch.cat((feat2_dec_r_1, feat2_dec_r_2), 2)
        del feat2_dec_r_1
        del feat2_dec_r_2
        x3r_img = torch.cat((x3r_img, feat2_dec_r), 1)
        x3l_img = torch.cat((x3l_img, feat2_dec_l), 1)
        del feat2_dec_l
        del feat2_dec_r


        # stage3 decoder
        feat3_enc_l = self.unet_enconder_patch2(x3l_img)
        feat3_enc_r = self.unet_enconder_patch2(x3r_img)
        del x3l_img
        del x3r_img
        feat3_enc_l[-1] = torch.cat((feat3_enc_l[-1], feat2_skip_l), 1)
        feat3_enc_r[-1] = torch.cat((feat3_enc_r[-1], feat2_skip_r), 1)
        del feat2_skip_l
        del feat2_skip_r
        feat3_enc = [torch.cat((k, v), 3) for k, v in zip(feat3_enc_l, feat3_enc_r)]
        del feat3_enc_l
        del feat3_enc_r
        feat3_skip = feat3_enc[-1]
        W3 = int(W/2/2**5)
        feat3_skip_1 = feat3_skip[:, :, :, 0:W3]
        feat3_skip_2 = feat3_skip[:, :, :, W3+ over_pix_1 * 2:]
        feat3_skip_1[:, :, :, W3- over_pix_1:] = (feat3_skip_1[:, :, :, W3- over_pix_1:] + feat3_skip[:, :, :, W3 + over_pix_1:W3 + over_pix_1 * 2]) / 2
        feat3_skip_2[:, :, :, 0:over_pix_1] = (feat3_skip_2[:, :, :, 0:over_pix_1] + feat3_skip[:, :, :, W3:W3 + over_pix_1]) / 2
        feat3_skip = torch.cat((feat3_skip_1, feat3_skip_2), 3)



        del feat3_skip_1
        del feat3_skip_2
        feat3_dec = self.unet_deconder_patch2(feat3_enc)
        W4 = int(W / 2)
        feat3_dec_1 = feat3_dec[:, :, :, 0:W4]
        feat3_dec_2 = feat3_dec[:, :, :, W4 + self.over_pix*2:]
        feat3_dec_1[:, :, :, W4 - self.over_pix:] = (feat3_dec_1[:, :, :, W4 - self.over_pix:] + feat3_dec[:, :, :, W4 + self.over_pix:W4 + self.over_pix*2]) / 2
        feat3_dec_2[:, :, :, 0:self.over_pix] = (feat3_dec_1[:, :, :, 0:self.over_pix] + feat3_dec[:, :, :, W4:W4 + self.over_pix]) / 2
        feat3_dec = torch.cat((feat3_dec_1, feat3_dec_2), 3)
        x_cros = torch.cat((x, feat3_dec), 1)
        del feat3_dec
        del feat3_dec_1
        del feat3_dec_2

        # stage4 encoder
        x_feat_enc = self.unet_enconder(x_cros)
        x_feat_enc[-1] = torch.cat((x_feat_enc[-1], feat3_skip), 1)
        del feat3_skip
        # stage4 decoder
        x_feat_dec = self.unet_deconder(x_feat_enc)
        del x_feat_enc

        x_adnet = self.adnet(x_cros)
        del x_cros
        xx = torch.cat((x_feat_dec, x_adnet), 1)
        del x_feat_dec
        xx = self.cov(xx)
        return xx


class CMBFSCNN_level4(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, n_feats=16):
        super(CMBFSCNN_level4, self).__init__()
        self.adnet = BRDNet(n_feats=n_feats, in_channel = in_channels*2)
        self.unet_enconder_patch1 = Encoder(in_channels = in_channels)
        self.unet_deconder_patch1 = Decoder(n_feat = 512, out_channels = in_channels)
        self.unet_enconder_patch2 = Encoder(in_channels = 2 *in_channels)
        self.unet_deconder_patch2 = Decoder(n_feat=512*2, out_channels = in_channels)
        self.unet_enconder_patch3 = Encoder(in_channels=2 * in_channels)
        self.unet_deconder_patch3 = Decoder(n_feat=512 * 3, out_channels = in_channels)
        self.unet_enconder = Encoder(in_channels=2 * in_channels)
        self.unet_deconder = Decoder(n_feat=512 * 4, out_channels = out_channels)
        self.cov = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, padding=1, stride=1))

    def forward(self, x):
        H = x.size(2)
        W = x.size(3)

        #  Patches for Stage 3
        x3l_img = x[:, :, :, 0:int(W / 2)+32]
        x3r_img = x[:, :, :, int(W / 2)-32:W]

        #  Patches for Stage 2
        x2ltop_img = x[:, :, 0:int(H / 2)+32, 0:int(W / 2)+32]
        x2rtop_img = x[:, :, 0:int(H / 2)+32, int(W / 2)-32:W]
        x2lbot_img = x[:, :, int(H / 2)-32:H, 0:int(W / 2)+32]
        x2rbot_img = x[:, :, int(H / 2)-32:H, int(W / 2)-32:W]

        #  Patches for Stage 1
        x1top1_img = x[:, :, 0:int(H / 2)+32, 0:int(W / 4)+32]
        x1top2_img = x[:, :, 0:int(H / 2)+32, int(W / 4)-32:int(W / 2)+32]
        x1top3_img = x[:, :, 0:int(H / 2)+32, int(W / 2)-32:int(3 * W / 4)+32]
        x1top4_img = x[:, :, 0:int(H / 2)+32, int(3 * W / 4)-32:W]

        x1bot1_img = x[:, :, int(H / 2)-32:H, 0:int(W / 4)+32]
        x1bot2_img = x[:, :, int(H / 2)-32:H, int(W / 4)-32:int(W / 2)+32]
        x1bot3_img = x[:, :, int(H / 2)-32:H, int(W / 2)-32:int(3 * W / 4)+32]
        x1bot4_img = x[:, :, int(H / 2)-32:H, int(3 * W / 4)-32:W]

        # stage1 encoder
        feat_enc_top1 = self.unet_enconder_patch1(x1top1_img)
        feat_enc_top2 = self.unet_enconder_patch1(x1top2_img)
        feat_enc_top3 = self.unet_enconder_patch1(x1top3_img)
        feat_enc_top4 = self.unet_enconder_patch1(x1top4_img)
        feat_enc_bot1 = self.unet_enconder_patch1(x1bot1_img)
        feat_enc_bot2 = self.unet_enconder_patch1(x1bot2_img)
        feat_enc_bot3 = self.unet_enconder_patch1(x1bot3_img)
        feat_enc_bot4 = self.unet_enconder_patch1(x1bot4_img)

        ## Concat deep features
        feat_enc_ltop = [torch.cat((k, v), 3) for k, v in zip(feat_enc_top1, feat_enc_top2)]
        feat_enc_rtop = [torch.cat((k, v), 3) for k, v in zip(feat_enc_top3, feat_enc_top4)]
        feat_enc_lbot = [torch.cat((k, v), 3) for k, v in zip(feat_enc_bot1, feat_enc_bot2)]
        feat_enc_rbot = [torch.cat((k, v), 3) for k, v in zip(feat_enc_bot3, feat_enc_bot4)]

        # ------------------- The boundary of the merged data for output of enconder of unit 1 ---------------
        feat1_skip_ltop = feat_enc_ltop[4]
        feat1_skip_ltop_1 = feat1_skip_ltop[:, :, :,0:4]
        feat1_skip_ltop_2 = feat1_skip_ltop[:, :, :,6:]
        feat1_skip_ltop_1[:, :, :,3] = (feat1_skip_ltop_1[:, :, :,3] + feat1_skip_ltop[:, :, :,5]) / 2
        feat1_skip_ltop_2[:, :, :, 0] = (feat1_skip_ltop_2[:, :, :,0] + feat1_skip_ltop[:, :, :,4]) / 2
        feat1_skip_ltop = torch.cat((feat1_skip_ltop_1, feat1_skip_ltop_2), 3)
        feat1_skip_rtop = feat_enc_rtop[4]
        feat1_skip_rtop_1 = feat1_skip_rtop[:, :, :, 0:4]
        feat1_skip_rtop_2 = feat1_skip_rtop[:, :, :, 6:]
        feat1_skip_rtop_1[:, :, :, 3] = (feat1_skip_rtop_1[:, :, :, 3] + feat1_skip_rtop[:, :, :, 5]) / 2
        feat1_skip_rtop_2[:, :, :, 0] = (feat1_skip_rtop_2[:, :, :, 0] + feat1_skip_rtop[:, :, :, 4]) / 2
        feat1_skip_rtop = torch.cat((feat1_skip_rtop_1, feat1_skip_rtop_2), 3)

        feat1_skip_lbot = feat_enc_lbot[4]
        feat1_skip_lbot_1 = feat1_skip_lbot[:, :, :, 0:4]
        feat1_skip_lbot_2 = feat1_skip_lbot[:, :, :, 6:]
        feat1_skip_lbot_1[:, :, :, 3] = (feat1_skip_lbot_1[:, :, :, 3] + feat1_skip_lbot[:, :, :, 5]) / 2
        feat1_skip_lbot_2[:, :, :, 0] = (feat1_skip_lbot_2[:, :, :, 0] + feat1_skip_lbot[:, :, :, 4]) / 2
        feat1_skip_lbot = torch.cat((feat1_skip_lbot_1, feat1_skip_lbot_2), 3)

        feat1_skip_rbot = feat_enc_rbot[4]
        feat1_skip_rbot_1 = feat1_skip_rbot[:, :, :, 0:4]
        feat1_skip_rbot_2 = feat1_skip_rbot[:, :, :, 6:]
        feat1_skip_rbot_1[:, :, :, 3] = (feat1_skip_rbot_1[:, :, :, 3] + feat1_skip_rbot[:, :, :, 5]) / 2
        feat1_skip_rbot_2[:, :, :, 0] = (feat1_skip_rbot[:, :, :, 0] + feat1_skip_rbot[:, :, :, 4]) / 2
        feat1_skip_rbot = torch.cat((feat1_skip_rbot_1, feat1_skip_rbot_2), 3)

        # stage1 decoder
        feat_dec_ltop = self.unet_deconder_patch1(feat_enc_ltop)
        feat_dec_rtop = self.unet_deconder_patch1(feat_enc_rtop)
        feat_dec_lbot = self.unet_deconder_patch1(feat_enc_lbot)
        feat_dec_rbot = self.unet_deconder_patch1(feat_enc_rbot)

        # ------------------- The boundary of the merged data for output of deconder of unit 1 ---------------
        feat_dec_ltop_1 = feat_dec_ltop[:, :, :,0:128]
        feat_dec_ltop_2 = feat_dec_ltop[:, :, :,128 + 64:]
        feat_dec_ltop_1[:, :, :,128-32:] = (feat_dec_ltop_1[:, :, :,128 - 32:] + feat_dec_ltop[:, :, :,128 + 32:128 + 64]) / 2
        feat_dec_ltop_2[:, :, :,0:32] = (feat_dec_ltop_1[:, :, :,0:32] + feat_dec_ltop[:, :, :,128:128 + 32]) / 2
        feat_dec_ltop = torch.cat((feat_dec_ltop_1, feat_dec_ltop_2), 3)

        feat_dec_rtop_1 = feat_dec_rtop[:, :, :, 0:128]
        feat_dec_rtop_2 = feat_dec_rtop[:, :, :, 128 + 64:]
        feat_dec_rtop_1[:, :, :, 128 - 32:] = (feat_dec_rtop_1[:, :, :, 128 - 32:] + feat_dec_rtop[:, :, :,128 + 32:128 + 64]) / 2
        feat_dec_rtop_2[:, :, :, 0:32] = (feat_dec_rtop_1[:, :, :, 0:32] + feat_dec_rtop[:, :, :, 128:128 + 32]) / 2
        feat_dec_rtop = torch.cat((feat_dec_rtop_1, feat_dec_rtop_2), 3)

        feat_dec_lbot_1 = feat_dec_lbot[:, :, :, 0:128]
        feat_dec_lbot_2 = feat_dec_lbot[:, :, :, 128 + 64:]
        feat_dec_lbot_1[:, :, :, 128 - 32:] = (feat_dec_lbot_1[:, :, :, 128 - 32:] + feat_dec_lbot[:, :, :,128 + 32:128 + 64]) / 2
        feat_dec_lbot_2[:, :, :, 0:32] = (feat_dec_lbot_1[:, :, :, 0:32] + feat_dec_lbot[:, :, :, 128:128 + 32]) / 2
        feat_dec_lbot = torch.cat((feat_dec_lbot_1, feat_dec_lbot_2), 3)

        feat_dec_rbot_1 = feat_dec_rbot[:, :, :, 0:128]
        feat_dec_rbot_2 = feat_dec_rbot[:, :, :, 128 + 64:]
        feat_dec_rbot_1[:, :, :, 128 - 32:] = (feat_dec_rbot_1[:, :, :, 128 - 32:] + feat_dec_rbot[:, :, :,128 + 32:128 + 64]) / 2
        feat_dec_rbot_2[:, :, :, 0:32] = (feat_dec_rbot_1[:, :, :, 0:32] + feat_dec_rbot[:, :, :, 128:128 + 32]) / 2
        feat_dec_rbot = torch.cat((feat_dec_rbot_1, feat_dec_rbot_2), 3)

        # ---------------------------------------------------------------------------------------------------
        x2ltop_img = torch.cat((x2ltop_img, feat_dec_ltop), 1)
        x2rtop_img = torch.cat((x2rtop_img, feat_dec_rtop), 1)
        x2lbot_img = torch.cat((x2lbot_img, feat_dec_lbot), 1)
        x2rbot_img = torch.cat((x2rbot_img, feat_dec_rbot), 1)

        # stage2 encoder
        feat2_enc_ltop = self.unet_enconder_patch2(x2ltop_img)
        feat2_enc_rtop = self.unet_enconder_patch2(x2rtop_img)
        feat2_enc_lbot = self.unet_enconder_patch2(x2lbot_img)
        feat2_enc_rbot = self.unet_enconder_patch2(x2rbot_img)
        feat2_enc_ltop[4] = torch.cat((feat2_enc_ltop[4], feat1_skip_ltop), 1)
        feat2_enc_rtop[4] = torch.cat((feat2_enc_rtop[4], feat1_skip_rtop),1)
        feat2_enc_lbot[4] = torch.cat((feat2_enc_lbot[4], feat1_skip_lbot), 1)
        feat2_enc_rbot[4] = torch.cat((feat2_enc_rbot[4], feat1_skip_rbot), 1)
        feat2_enc_l = [torch.cat((k, v), 2) for k, v in zip(feat2_enc_ltop, feat2_enc_lbot)]
        feat2_enc_r = [torch.cat((k, v), 2) for k, v in zip(feat2_enc_rtop, feat2_enc_rbot)]
        feat2_skip_l = feat2_enc_l[4]
        feat2_skip_l_1 = feat2_skip_l[:,:,0:8,:]
        feat2_skip_l_2 = feat2_skip_l[:, :, 10:, :]
        feat2_skip_l_1[:,:,7,:] = (feat2_skip_l_1[:,:,7,:] + feat2_skip_l[:,:,9,:])/2
        feat2_skip_l_2[:,:,0,:] = (feat2_skip_l_2[:,:,0,:] + feat2_skip_l[:,:,8,:])/2
        feat2_skip_l = torch.cat((feat2_skip_l_1, feat2_skip_l_2),2)
        # feat2_enc_l[4] = feat2_skip_l

        feat2_skip_r = feat2_enc_r[4]
        feat2_skip_r_1 = feat2_skip_r[:, :, 0:8, :]
        feat2_skip_r_2 = feat2_skip_r[:, :, 10:, :]
        feat2_skip_r_1[:, :, 7, :] = (feat2_skip_r_1[:, :, 7, :] + feat2_skip_r[:, :, 9, :]) / 2
        feat2_skip_r_2[:, :, 0, :] = (feat2_skip_r_2[:, :, 0, :] + feat2_skip_r[:, :, 8, :]) / 2
        feat2_skip_r = torch.cat((feat2_skip_r_1, feat2_skip_r_2), 2)
        # feat2_enc_r[4] = feat2_skip_r

        # stage2 decoder
        feat2_dec_l = self.unet_deconder_patch2(feat2_enc_l)
        feat2_dec_r = self.unet_deconder_patch2(feat2_enc_r)
        feat2_dec_l_1 = feat2_dec_l[:, :, 0:256, :]
        feat2_dec_l_2 = feat2_dec_l[:, :, 256 + 64:, :]
        feat2_dec_l_1[:, :, 256 - 32:, :] = (feat2_dec_l_1[:, :, 256 - 32:, :] + feat2_dec_l[:, :, 256 + 32:256 + 64,
                                                                                 :]) / 2
        feat2_dec_l_2[:, :, 0:32, :] = (feat2_dec_l_1[:, :, 0:32, :] + feat2_dec_l[:, :, 256:256 + 32, :]) / 2
        feat2_dec_l = torch.cat((feat2_dec_l_1, feat2_dec_l_2), 2)

        feat2_dec_r_1 = feat2_dec_r[:, :, 0:256, :]
        feat2_dec_r_2 = feat2_dec_r[:, :, 256 + 64:, :]
        feat2_dec_r_1[:, :, 256 - 32:, :] = (feat2_dec_r_1[:, :, 256 - 32:, :] + feat2_dec_r[:, :, 256 + 32:256 + 64,
                                                                                 :]) / 2
        feat2_dec_r_2[:, :, 0:32, :] = (feat2_dec_r_1[:, :, 0:32, :] + feat2_dec_r[:, :, 256:256 + 32, :]) / 2
        feat2_dec_r = torch.cat((feat2_dec_r_1, feat2_dec_r_2), 2)
        x3r_img = torch.cat((x3r_img, feat2_dec_r), 1)
        x3l_img = torch.cat((x3l_img, feat2_dec_l), 1)

        # stage3 decoder
        feat3_enc_l = self.unet_enconder_patch3(x3l_img)
        feat3_enc_r = self.unet_enconder_patch3(x3r_img)
        feat3_enc_l[4] = torch.cat((feat3_enc_l[4], feat2_skip_l), 1)
        feat3_enc_r[4] = torch.cat((feat3_enc_r[4], feat2_skip_r), 1)
        feat3_enc = [torch.cat((k, v), 3) for k, v in zip(feat3_enc_l, feat3_enc_r)]
        feat3_skip = feat3_enc[4]
        feat3_skip_1 = feat3_skip[:, :, :, 0:8]
        feat3_skip_2 = feat3_skip[:, :, :, 10:]
        feat3_skip_1[:, :, :, 7] = (feat3_skip_1[:, :, :, 7] + feat3_skip[:, :, :, 9]) / 2
        feat3_skip_2[:, :, :, 0] = (feat3_skip_2[:, :, :, 0] + feat3_skip[:, :, :, 8]) / 2
        feat3_skip = torch.cat((feat3_skip_1, feat3_skip_2), 3)
        feat3_dec = self.unet_deconder_patch3(feat3_enc)
        feat3_dec_1 = feat3_dec[:, :, :, 0:256]
        feat3_dec_2 = feat3_dec[:, :, :, 256 + 64:]
        feat3_dec_1[:, :, :, 256 - 32:] = (feat3_dec_1[:, :, :, 256 - 32:] + feat3_dec[:, :, :, 256 + 32:256 + 64]) / 2
        feat3_dec_2[:, :, :, 0:32] = (feat3_dec_1[:, :, :, 0:32] + feat3_dec[:, :, :, 256:256 + 32]) / 2
        feat3_dec = torch.cat((feat3_dec_1, feat3_dec_2), 3)
        x_cros = torch.cat((x, feat3_dec), 1)
        # stage4 encoder
        x_feat_enc = self.unet_enconder(x_cros)
        x_feat_enc[4] = torch.cat((x_feat_enc[4], feat3_skip), 1)
        # stage4 decoder
        x_feat_dec = self.unet_deconder(x_feat_enc)

        x_adnet = self.adnet(x_cros)
        xx = torch.cat((x_feat_dec, x_adnet), 1)
        xx = self.cov(xx)
        return xx



class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()
        # the shape of input: (B, 1, nside*4, nside*3) = (B, 1, 2048, 1536)
        self.down1 = UNetDown(in_channels, 16, normalize=False,kernel_size=4,stride=2,padding=1) # (N,64,L/2,H/2)
        self.down2 = UNetDown(16, 32, kernel_size=4,stride=2,padding=1)  # (N,128,L/2**2,H/2**2)
        self.down3 = UNetDown(32, 64,kernel_size=4,stride=2,padding=1)  # (N,256,L/2**3,H/2**3)
        self.down4 = UNetDown(64, 128, kernel_size=4,stride=2,padding=1) # (N,512,L/2**4,H/2**4)
        self.down5 = UNetDown(128, 256, kernel_size=4, stride=2, padding=1)  # (N,512,L/2**5,H/2**5)
        # self.down6 = UNetDown(256, 256, kernel_size=3, stride=1, padding=1)  # (N,512,L/2**5,H/2**5)


        # self.up3 = UNetUp(256, 256, kernel_size=3,stride=1,padding=1)  # (N,256*2,L/2**3,H/2**3)
        self.up4 = UNetUp(256, 128, kernel_size=4, stride=2, padding=1)  # (N,128*2,L/2**2,H/2**2)
        self.up5 = UNetUp(256, 64, kernel_size=4, stride=2, padding=1)  # (N,64*2,L/2,H/2)
        self.up6 = UNetUp(128, 32, kernel_size=4, stride=2, padding=1)  # (N,64*2,L/2,H/2)
        self.up7 = UNetUp(64, 16, kernel_size=4, stride=2, padding=1)  # (N,64*2,L/2,H/2)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.4, inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.4, inplace=True),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u4 = self.up4(d5, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.up8(u7)
        uf = self.final(u8)
        return uf




class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,padding=1,stride=1,normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size,stride=stride, padding = padding, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.4, inplace=True))
        if dropout >0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, kernel_size=3,padding=1,stride=1):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride, padding = padding, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.4, inplace=True),
        ]
        if dropout >0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x





class Encoder(nn.Module):
    def __init__(self, in_channels=6):
        super(Encoder, self).__init__()
        self.down1 = UNetDown(in_channels, 16, kernel_size=4, stride=2, padding=1)
        self.down2 = UNetDown(16, 32, kernel_size=4, stride=2, padding=1)
        self.down3 = UNetDown(32, 64, kernel_size=4, stride=2, padding=1)
        self.down4 = UNetDown(64, 128, kernel_size=4, stride=2, padding=1)
        self.down5 = UNetDown(128, 256, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        return [d1, d2, d3, d4, d5]

class Decoder(nn.Module):
    def __init__(self, out_channels=6, n_feat=256):
        super(Decoder, self).__init__()
        self.up1 = UNetUp(n_feat, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = UNetUp(256, 64, kernel_size=4, stride=2, padding=1)
        self.up3 = UNetUp(128, 32, kernel_size=4, stride=2, padding=1)
        self.up4 = UNetUp(64, 16, kernel_size=4, stride=2, padding=1)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            #nn.InstanceNorm2d(16),
            #nn.LeakyReLU(0.4, inplace=True),
        )

    def forward(self, x):
        d1, d2, d3, d4, d5 = x
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4)
        return u5



class UpNet(nn.Module):

    def __init__(self,n_feats,in_channel):
        super(UpNet, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channel, out_channels=n_feats, kernel_size=3, stride=1, padding=1),
                # BatchRenorm2d(64),
                nn.InstanceNorm2d(n_feats),
                nn.LeakyReLU(0.4, inplace=True)]

        for i in range(5):
            layers.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1))
            # layers.append(BatchRenorm2d(64))
            layers.append(nn.InstanceNorm2d(n_feats))
            layers.append(nn.LeakyReLU(0.4, inplace=True))

        layers.append(nn.Conv2d(n_feats, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class DownNet(nn.Module):
    def __init__(self,n_feats,in_channel):
        super(DownNet, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channel, out_channels=n_feats, kernel_size=3, stride=1, padding=1),
                #BatchRenorm2d(64),
                nn.InstanceNorm2d(n_feats),
                nn.LeakyReLU(0.4, inplace=True)]

        for i in range(2):
            layers.append(nn.Conv2d(n_feats, n_feats, 3, 1, padding=2, dilation=2))
            layers.append(nn.LeakyReLU(0.4, inplace=True))
        layers.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1))
        #layers.append(BatchRenorm2d(64))
        layers.append(nn.InstanceNorm2d(n_feats))
        layers.append(nn.LeakyReLU(0.4, inplace=True))
        for i in range(2):
            layers.append(nn.Conv2d(n_feats, n_feats, 3, 1, padding=2, dilation=2))
            layers.append(nn.LeakyReLU(0.4, inplace=True))
        layers.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1))
        # layers.append(BatchRenorm2d(64))
        layers.append(nn.InstanceNorm2d(n_feats))
        layers.append(nn.LeakyReLU(0.4, inplace=True))

        layers.append(nn.Conv2d(n_feats, 1, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class BRDNet(nn.Module):
    def __init__(self,n_feats, in_channel):
        super(BRDNet, self).__init__()
        self.upnet = UpNet(n_feats=n_feats,in_channel=in_channel)
        self.dwnet = DownNet(n_feats=n_feats,in_channel=in_channel)
        self.conv = nn.Conv2d(in_channel+2, 1, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #import pdb;pdb.set_trace()
        out1 = self.upnet(x)
        out2 = self.dwnet(x)
        #out1 = x - out1
        #out2 = x - out2
        out = torch.cat((out1, out2, x), 1)
        out = self.conv(out)
        #out = x - out
        return out



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)




