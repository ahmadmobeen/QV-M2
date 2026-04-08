import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse

# Enable loading argparse.Namespace in PyTorch 2.6+
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace])

from FlashMMR.config import TestOptions
from FlashMMR.start_end_dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from FlashMMR.postprocessing import PostProcessorDETR
from standalone_eval.eval import eval_submission
from utils.basic_utils import save_jsonl, save_json

import nncore
from nncore.ops import temporal_iou

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms, nms_type):
    mr_res_after_nms = []
    for e in mr_res:
        bnd = torch.tensor(e["pred_relevant_windows"])
        for i in range(bnd.size(0)):
            max_idx = bnd[i:, -1].argmax(dim=0)
            bnd = nncore.swap_element(bnd, i, max_idx + i)
            iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

            if nms_type == 'normal':
                bnd[i + 1:, -1][iou >= nms_thd] = 0
            elif nms_type == 'linear':
                bnd[i + 1:, -1] *= 1 - iou
            else:
                raise ValueError(f"Unknown nms_type: {nms_type}")

        _, inds = bnd[:, -1].sort(descending=True)
        bnd = bnd[inds]
        e["pred_relevant_windows"] = bnd.tolist()

        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data, verbose=opt.debug, match_number=not opt.debug
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [
            submission_path,
        ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission,
            nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms,
            max_after_nms=opt.max_after_nms,
            nms_type=opt.nms_type,
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(
            ".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd)
        )
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms,
                gt_data,
                verbose=opt.debug,
                match_number=not opt.debug,
            )
            save_metrics_nms_path = submission_nms_path.replace(
                ".jsonl", "_metrics.json"
            )
            save_json(
                metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False
            )
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [
                submission_nms_path,
            ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths

# for HL
@torch.no_grad()
def compute_hl_results(
    model, eval_loader, opt, epoch_i=None, tb_writer=None
):
    model.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []

    topk = 5  # top-5 map

    video_ap_collected = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is not None:
            targets["label"] = batch[0]
            targets["fps"] = torch.full((256,), 1/opt.clip_length).to(opt.device) # if datasets is qv, fps is 0.5, charades' is 1
        else:
            targets = {}

        outputs = model(**model_inputs, targets=targets)

        preds = outputs["saliency_scores"].clone().detach()

        for meta, pred in zip(query_meta, preds):
            pred = pred
            label = meta["label"]  # raw label

            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"

            if opt.dset_name in ["tvsum"]:
                for i in range(20):
                    pred = pred.cpu()
                    cur_pred = pred[: len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, 0.0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)

            elif opt.dset_name in ["youtube_uni"]:
                cur_pred = pred[: len(label)]
                # if opt.dset_name == "tvsum_sfc":
                cur_pred = cur_pred.cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)

                cur_label = torch.Tensor(label).squeeze()[inds].tolist()

                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc

                video_ap.append(float(ap))
            else:
                print("No such dataset")
                exit(-1)

            video_ap_collected.append(video_ap)

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))

    # tensorboard writer
    if write_tb:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return submmission, loss_meters

# for MR
@torch.no_grad()
def compute_mr_results(
    model, eval_loader, opt, epoch_i=None, tb_writer=None
):
    model.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is not None:
            targets["label"] = batch[0]
            targets["fps"] = torch.full((256,), 1/opt.clip_length).to(opt.device) # if datasets is qv, fps is 0.5, charades' is 1
        else:
            targets = {}
        outputs = model(**model_inputs, targets=targets)

        # Batched Pyramid Decoding (VMR / Moment Retrieval)
        if opt.span_loss_type == "l1":
            all_saliency = outputs["saliency_scores"] # [bs, L]
            all_class = outputs["out_class"].sigmoid() # [bs, num_pts, 1]
            all_coord = outputs["out_coord"].clone() # [bs, num_pts, 2]
            all_points = outputs["point"] # [bs, num_pts, 4]
            
            # Apply Decoding Logic (Scaling by clip_length)
            all_coord[:, :, 0] *= -1
            all_coord *= all_points[:, :, 3, None].repeat(1, 1, 2)
            all_coord += all_points[:, :, 0, None].repeat(1, 1, 2)
            all_coord /= (1.0 / opt.clip_length)
            
            # Prep per-sample results
            bsz = all_saliency.shape[0]
            for idx in range(bsz):
                meta = query_meta[idx]
                
                # 1. Saliency
                ss = all_saliency[idx].cpu().tolist()
                v_mask = model_inputs["src_vid_mask"][idx]
                actual_len = int(v_mask.sum().item())
                ss = [float(f"{e:.4f}") for e in ss[:actual_len]]
                
                # 2. Windows
                scores, _ = all_class[idx].max(-1)
                cur_coord = all_coord[idx]
                # Clamp to duration
                cur_coord = torch.clamp(cur_coord, 0, meta["duration"])
                
                valid_windows = torch.cat((cur_coord, scores.unsqueeze(-1)), dim=-1)
                # Sort by score
                _, inds = scores.sort(descending=True)
                # Take all windows for post-processor to handle NMS
                cur_ranked_preds = valid_windows[inds].tolist()
                
                cur_query_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_relevant_windows=cur_ranked_preds,
                    pred_saliency_scores=ss,
                )
                mr_res.append(cur_query_pred)
        else:
            # Traditional span classification logic (if used)
            bsz = outputs["pred_spans"].shape[0]
            # ... (omitted for brevity)
            pass

        loss_dict = {k: v for k, v in outputs.items() if 'loss' in k}
        losses = sum(loss_dict.values())
        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v)
            )

    if write_tb and len(loss_meters) != 1:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    if opt.dset_name in ["hl"]:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length,
            min_ts_val=0,
            max_ts_val=150,
            min_w_l=2,
            max_w_l=150,
            move_window_method="left",
            process_func_names=("clip_ts", "round_multiple"),
        )
    elif opt.dset_name in ["charadesSTA"]:
        if opt.v_feat_dim == 4096:  # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length,
                min_ts_val=0,
                max_ts_val=360,
                min_w_l=12,
                max_w_l=360,
                move_window_method="left",
                process_func_names=("clip_ts", "round_multiple"),
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length,
                min_ts_val=0,
                max_ts_val=150,
                min_w_l=2,
                max_w_l=60,
                move_window_method="left",
                process_func_names=("clip_ts", "round_multiple"),
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length,
            min_ts_val=0,
            max_ts_val=50000,
            min_w_l=0,
            max_w_l=50000,
            move_window_method="left",
            process_func_names=(["round_multiple"]),
        )

    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


def get_eval_res(model, eval_loader, opt, epoch_i, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(
        model, eval_loader, opt, epoch_i, tb_writer
    )  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(
    model,
    eval_dataset,
    opt,
    save_submission_filename,
    epoch_i=None,
    tb_writer=None,
):
    logger.info("Generate submissions")
    model.eval()

    if opt.dset_name == "tacos":
        shuffle = True
    else:
        shuffle = False

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=shuffle,
        pin_memory=opt.pin_memory,
    )

    # tvsum
    if opt.dset_name in ["tvsum", "youtube_uni"]:
        metrics, eval_loss_meters = compute_hl_results(
            model, eval_loader, opt, epoch_i, tb_writer
        )

        # to match original save format
        submission = [{"brief": metrics}]
        submission_path = os.path.join(opt.results_dir, "latest_metric.jsonl")
        save_jsonl(submission, submission_path)

        return submission[0], submission[0], eval_loss_meters, [submission_path]

    else:
        submission, eval_loss_meters = get_eval_res(
            model, eval_loader, opt, epoch_i, tb_writer
        )

        if opt.dset_name in ["charadesSTA", "tacos", "nlq"]:
            new_submission = []
            for s in submission:
                s.pop("pred_saliency_scores", None)
                new_submission.append(s)
            submission = new_submission

        metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
            submission, opt, eval_dataset.data, save_submission_filename
        )
        return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model")
    from FlashMMR.model import build_model
    model = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    if opt.resume_adapter is not None:
        logger.info(f"Load adapter checkpoint from {opt.resume_adapter}")
        adapter_checkpoint = torch.load(opt.resume_adapter, weights_only=False)
        adapter_state_dict = {k: v for k, v in adapter_checkpoint['state_dict'].items() if k.startswith('adapter')}
        model.load_state_dict(adapter_state_dict, strict=False)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=False)

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        if "pt" in opt.resume[:-4]:
            if "asr" in opt.resume[:25]:
                model.load_state_dict(checkpoint["model"])
            else:
                for k, v in checkpoint["state_dict"].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # model.load_state_dict(checkpoint["model"])
                model.load_state_dict(new_state_dict)
        else:
            # model.load_state_dict(checkpoint["state_dict"])
            model.load_state_dict(checkpoint["model"], strict=False)
    else:
        logger.warning(
            "If you intend to evaluate the model, please specify --resume with ckpt path"
        )

    return model


def start_inference(train_opt=None, split=None, splitfile=None):
    if train_opt is not None:
        opt = TestOptions().parse(train_opt.a_feat_dir)
    else:
        opt = TestOptions().parse()
    if split is not None:
        opt.eval_split_name = split
    if splitfile is not None:
        opt.eval_path = splitfile

    opt.cfg = nncore.Config.from_file(opt.config)

    print(opt.eval_split_name)
    print(opt.eval_path)
    logger.info("Setup config, data and model...")

    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    if opt.eval_split_name == "val":
        loadlabel = True
    else:
        loadlabel = False

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=loadlabel,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
    )
    model = setup_model(opt)
    save_submission_filename = "hl_{}_submission.jsonl".format(opt.eval_split_name)

    logger.info("FORCING model.train() for parity check...")
    model.train()
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = eval_epoch(
            model, eval_dataset, opt, save_submission_filename
        )
    if opt.eval_split_name == "val":
        logger.info(
            "metrics_no_nms {}".format(
                pprint.pformat(metrics_no_nms["brief"], indent=4)
            )
        )
    if metrics_nms is not None:
        logger.info(
            "metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4))
        )


from sys import argv

if __name__ == "__main__":
    start_inference()
