import  os
import  torch
import  tqdm
import  numpy as np
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import cv2

from util import deccode_output_score_and_ptss, seg_point
from utils.logger import TxtLogger
from utils.meter import AverageMeter

from utils.decode import deccode_lines_TP
from loss import LineSegmentLoss
from metric import F1_score_128, msTPFP, AP
import torch.nn as nn
# from apex.fp16_utils import *
# from apex import amp, optimizers
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size()
        num = predict.size(0)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score
class Simple_MLSD_Learner():
    def __init__(self,
                 cfg,
                 model : torch.nn.Module,
                 optimizer: Optimizer,
                 scheduler,
                 logger : TxtLogger,
                 save_dir : str,
                 log_steps = 100,
                 device_ids = [0,1],
                 gradient_accum_steps = 1,
                 max_grad_norm = 100.0,
                 batch_to_model_inputs_fn = None,
                 early_stop_n = 4,
                 ):
        self.cfg = cfg
        self.model  = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.logger = logger
        self.device_ids = device_ids
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn  = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        self.global_step = 0

        self.input_size = self.cfg.datasets.input_size
        self.loss_fn = LineSegmentLoss(cfg)
        self.epo = 0
        self.diceloss=DiceLoss()


    def step(self,step_n,  batch_data : dict):
        imgs  = batch_data["xs"].cuda()
        label = batch_data["ys"].cuda()
        seg=batch_data["seg"].cuda()
        # output = torch.squeeze(imgs, dim=0)
        # output = output.permute(1, 2, 0) * 255
        # output = output.cpu().data.numpy()
        # #output = np.where(output > 125, 255, 0)
        # output = output.astype(np.uint8)
        # output = cv2.resize(output, (720, 440))
        # cv2.imshow("output",output)
        # cv2.waitKey(0)


        outputs ,seg_output= self.model(imgs)


        loss_dict = self.loss_fn(outputs, label,batch_data["gt_lines_tensor_512_list"],batch_data["sol_lines_512_all_tensor_list"])
        #loss_seg=self.diceloss(outputs[:,15,:,:],seg[:,0,:,:])
        loss_seg=self.diceloss(seg_output,seg)

        loss = 0.5*loss_dict['loss']+0.5*loss_seg

        if self.gradient_accum_steps > 1:
            loss = loss / self.gradient_accum_steps

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (step_n + 1) % self.gradient_accum_steps == 0:
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        return loss, loss_dict,loss_seg


    def val(self, model, val_dataloader : DataLoader):
        thresh = self.cfg.decode.score_thresh
        topk = self.cfg.decode.top_k
        min_len = self.cfg.decode.len_thresh

        model = model.eval()
        sap_thresh = 10
        data_iter = tqdm.tqdm(val_dataloader)
        f_scores = []
        recalls = []
        precisions = []

        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0

        for batch_data in data_iter:
            imgs = batch_data["xs"].cuda()
            label = batch_data["ys"].cuda()
            batch_outputs,seg_output = model(imgs)

            # keep TP mask
            label = label[:, 7:, :, :]
            batch_outputs = batch_outputs[:, 7:14, :, :]

#             batch_outputs[:, 0, :, :] = label[:, 0, :, :]
#             batch_outputs[:, 1:5, :,: ] = label[:, 1:5, :,: ]


            for outputs, gt_lines_512 in zip(batch_outputs, batch_data["gt_lines_512"]):
                gt_lines_512 = np.array(gt_lines_512, np.float32)

                outputs = outputs.unsqueeze(0)
                #pred_lines,scores = deccode_lines(outputs, thresh, min_len, topk, 3)

                center_ptss, pred_lines, _, scores = deccode_lines_TP(outputs, thresh, min_len, topk, 3)

                #print('pred_lines: ', pred_lines.shape)
                #print('gt_lines_512: ', gt_lines_512.shape)
                pred_lines =pred_lines.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()

                pred_lines_128 = 128 * pred_lines / (self.input_size / 2)

                gt_lines_128 = gt_lines_512 / 4
                fscore, recall, precision = F1_score_128(pred_lines_128.tolist(),gt_lines_128.tolist(),thickness=3)
                f_scores.append(fscore)
                recalls.append(recall)
                precisions.append(precision)

                tp, fp = msTPFP(pred_lines_128, gt_lines_128, sap_thresh)

                n_gt += gt_lines_128.shape[0]
                tp_list.append(tp)
                fp_list.append(fp)
                scores_list.append(scores)

        f_score = np.array(f_scores, np.float32).mean()
        recall = np.array(recalls, np.float32).mean()
        precision = np.array(precisions, np.float32).mean()


        tp_list = np.concatenate(tp_list)
        fp_list = np.concatenate(fp_list)
        scores_list = np.concatenate(scores_list)
        idx = np.argsort(scores_list)[::-1]
        tp = np.cumsum(tp_list[idx]) / n_gt
        fp = np.cumsum(fp_list[idx]) / n_gt
        rcs = tp
        pcs = tp / np.maximum(tp + fp, 1e-9)
        sAP = AP(tp, fp) * 100
        self.logger.write("==>step: {}, f_score: {}, recall: {}, precision:{}, sAP10: {}\n ".format(self.global_step, f_score, recall, precision, sAP))


        return {
            'fscore': f_score,
            'recall': recall,
            'precision':precision,
            'sAP10': sAP
        }

    # def val(self, model, val_dataloader: DataLoader):
    #     score_thr = 0.1
    #     dist_thr = 20
    #
    #     model = model.eval()
    #     data_iter = tqdm.tqdm(val_dataloader)
    #     precision = 0
    #     size=val_dataloader.__len__()
    #
    #     for batch_data in data_iter:
    #         imgs = batch_data["xs"].cuda()
    #         label = batch_data["seg"].cuda()
    #
    #         batch_outputs, seg_output = model(imgs)
    #
    #         # keep TP mask
    #         batch_outputs = batch_outputs[:, 7:, :, :]
    #         pre = 0
    #         for outputs, gt in zip(batch_outputs, label):
    #             outputs=torch.unsqueeze(outputs, dim=0)
    #             gt = torch.unsqueeze(gt, dim=0)
    #             pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    #             start = vmap[:, :, :2]
    #             end = vmap[:, :, 2:]
    #             dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))
    #
    #             segments_list = []
    #             for center, score in zip(pts, pts_score):
    #                 y, x = center
    #                 distance = dist_map[y, x]
    #                 if score > score_thr and distance > dist_thr:
    #                     disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
    #                     x_start = x + disp_x_start
    #                     y_start = y + disp_y_start
    #                     x_end = x + disp_x_end
    #                     y_end = y + disp_y_end
    #                     segments_list.append([x_start, y_start, x_end, y_end])
    #
    #             lines = 2 * np.array(segments_list)
    #
    #             seg_points = seg_point(gt)
    #
    #             result_seg = []
    #             len_seg = 1000
    #             for line_end in lines:
    #                 # 取和分割最近的线
    #                 len = np.sqrt((line_end[0] - seg_points[0]) ** 2 + (line_end[1] - seg_points[1]) ** 2)
    #                 if (len < len_seg):
    #                     len_seg = len
    #                     result_seg = [line_end[0],line_end[1],line_end[2],line_end[3]]
    #             if (seg_points == [0, 0] or result_seg==[]):
    #                 result——ablation = [0, 0]
    #             else:
    #                 result——ablation = result_seg
    #
    #             pre += np.sqrt((result——ablation[0] - seg_points[0]) ** 2 + (result——ablation[1] - seg_points[1]) ** 2)
    #         precision += pre / batch_outputs.shape[0]
    #
    #     self.logger.write("==>step: {}, precision:{}\n ".format(self.global_step, precision / size))
    #
    #     return {'precision': precision / size,}


    def train(self, train_dataloader : DataLoader,val_dataloader : DataLoader,epoches = 100):
        best_score = 0
        early_n = 0
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
        #                                             opt_level="O1",
        #                                             loss_scale=1.0
        #                                             )
        for self.epo in range(epoches):
            step_n = 0
            train_avg_loss = AverageMeter()
            train_avg_center_loss = AverageMeter()
            train_avg_replacement_loss = AverageMeter()
            train_avg_line_seg_loss = AverageMeter()

            train_avg_match_loss = AverageMeter()
            train_avg_match_rario = AverageMeter()
            train_avg_seg_loss = AverageMeter()

            data_iter = tqdm.tqdm(train_dataloader)
            for batch in data_iter:
                self.model.train()
                train_loss,loss_dict,loss_seg = self.step(step_n, batch)

                train_avg_loss.update(train_loss.item(),1)
                train_avg_seg_loss.update(loss_seg.item(),1)

                train_avg_center_loss.update(loss_dict['center_loss'].item() ,1)
                train_avg_replacement_loss.update(loss_dict['displacement_loss'].item(), 1)
                train_avg_line_seg_loss.update(loss_dict['line_seg_loss'].item(),1)
                train_avg_match_loss.update(float(loss_dict['match_loss']), 1)
                train_avg_match_rario.update(loss_dict['match_ratio'], 1)

                status = '[{0}] lr= {1:.6f} loss= {2:.3f} avg = {3:.3f},c: {4:.3f}, d: {5:.3f}, l: {6:.3f}, ' \
                         'm:{7:.3f},m_r:{8:.2f} ,seg:{9:.2f}'.format(
                    self.epo + 1,
                    self.scheduler.get_lr()[0],
                    train_loss.item(),
                    train_avg_loss.avg,
                    train_avg_center_loss.avg,
                    train_avg_replacement_loss.avg,


                    train_avg_line_seg_loss.avg,
                    train_avg_match_loss.avg,
                    train_avg_match_rario.avg,
                    train_avg_seg_loss.avg,
                    )

                #if step_n%self.log_steps ==0:
                #    print(status)
                data_iter.set_description(status)
                step_n +=1

            ##self.scheduler.step() ## we update every step instead
            if self.epo > self.cfg.val.val_after_epoch:
                self.model.eval()
                ## val
                m = self.val(self.model, val_dataloader)
                fscore = m['sAP10']
                if fscore>best_score:
                    early_n = 0
                    best_score = fscore
                    model_path = os.path.join(self.save_dir, 'best.pth')
                    torch.save(self.model.state_dict(), model_path)
                else:
                    early_n += 1
                self.logger.write("epo: {}, steps: {} ,sAP10 : {:.4f} , best sAP10: {:.4f}". format(self.epo, self.global_step, fscore, best_score))
                self.logger.write(str(m))
                self.logger.write("=="*50)

                if early_n > self.early_stop_n:
                    print('early stopped!')
                    return best_score
            model_path = os.path.join(self.save_dir, 'latest.pth')
            torch.save(self.model.state_dict(), model_path)
        return  best_score



