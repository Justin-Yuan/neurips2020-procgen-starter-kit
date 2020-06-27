import os 
import sys 
import math 
import logging 

from ray.tune.logger import Logger, VALID_SUMMARY_TYPES
from ray.tune.utils import flatten_dict
from ray.tune.result import (TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL)
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

import logging
logger = logging.getLogger(__name__)


##########################################################################################
####################################     File Utils   ##################################
##########################################################################################

def mkdirs(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)



##########################################################################################
####################################     Model Utils   ##################################
#########################################################################################

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_conv_output_shape(in_shape, kernel, stride=1, padding=0, dilation=1, out_channels=32):
    """ infer output shape after 1 layer of conv 
    reference: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    """
    _, h, w = in_shape
    out_h = math.floor(
        (h + 2 * padding - dilation * (kernel - 1) - 1) / float(stride) + 1) 
    out_w = math.floor(
        (w + 2 * padding - dilation * (kernel - 1) - 1) / float(stride) + 1) 
    return (out_channels, out_h, out_w) 



##########################################################################################
####################################     Logger   ##################################
##########################################################################################


# from https://github.com/ray-project/ray/blob/master/python/ray/tune/logger.py
class TBXLogger(Logger):
    """TensorBoard Logger.
    Automatically flattens nested dicts to show on TensorBoard:
        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
    """

    def _init(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.error("Upgrade to the latest pytorch.")
            raise
        self._file_writer = SummaryWriter(self.logdir, flush_secs=10)
        self.last_result = None

    def on_result(self, result):
        
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        flat_result = flatten_dict(tmp, delimiter="/")
        path = ["ray", "tune"]
        valid_result = {
            "/".join(path + [attr]): value
            for attr, value in flat_result.items()
            if type(value) in VALID_SUMMARY_TYPES
        }

        for attr, value in valid_result.items():
            self._file_writer.add_scalar(attr, value, global_step=step)
        self.last_result = valid_result
        self._file_writer.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            self._file_writer.close()



def build_TBXLogger(*args):
    """ factory function for building tensorboard logger with filtered loggings 
    """
    all_fields = [
        'episode_reward_max', 
        'episode_reward_min', 
        'episode_reward_mean', 
        'episode_len_mean', 
        'episodes_this_iter', 
        'policy_reward_min', 
        'policy_reward_max', 
        'policy_reward_mean', 
        'custom_metrics', 
        'sampler_perf', 
        'off_policy_estimator', 
        'info', 
        'timesteps_this_iter', 
        'done', 
        'timesteps_total', 
        'episodes_total', 
        'training_iteration', 
        'experiment_id', 
        'date', 
        'timestamp', 
        'time_this_iter_s', 
        'time_total_s', 
        'pid', 
        'hostname', 
        'node_ip', 
        'config', 
        'time_since_restore', 
        'timesteps_since_restore', 
        'iterations_since_restore'
    ]

    class FilteredTBXLogger(TBXLogger):
        """ modifications  
        - with modified `on_result` method to log out only required fields 
        - with `on_result` also logs rollout frames 
        """
        keep_fields = args
        log_videos = True 
        fps = 4 
        log_sys_usage = True 

        def on_result(self, result):
            step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
            tmp = result.copy()
            for k in [
                    "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
            ]:
                if k in tmp:
                    del tmp[k]  # not useful to log these

            # log system usage
            perf = result.get("perf", None)
            if FilteredTBXLogger.log_sys_usage and perf is not None:
                self.log_system_usage(step, perf)

            flat_result = flatten_dict(tmp, delimiter="/")
            path = ["scalars"]
            valid_result = {
                "/".join(path + [attr]): value
                for attr, value in flat_result.items()
                if type(value) in VALID_SUMMARY_TYPES 
                and attr in FilteredTBXLogger.keep_fields
            }

            # log scalars 
            for attr, value in valid_result.items():
                self._file_writer.add_scalar(attr, value, global_step=step)

            # log videos 
            videos = result.get("eval_frames", [])
            if FilteredTBXLogger.log_videos and len(videos) > 0:
                self.log_videos(step, videos, "rollout_frames")

            self.last_result = valid_result
            self._file_writer.flush()

        def log_images(self, step, images, tag):
            """ show images on tb, e.g. sequential graph structures, images: (N,C,H,W) """ 
            self._file_writer.add_images(tag, images, global_step=step)

        def log_videos(self, step, videos, tag):
            """ show rollouts on tb, videos: (T,H,W,C) """
            t, h, w, c = videos.shape
            # tb accepts (N,T,C,H,W)
            vid_tensor = torch.as_tensor(videos).permute(0,3,1,2).reshape(-1,t,c,h,w)   
            self._file_writer.add_video(tag, vid_tensor, global_step=step, fps=FilteredTBXLogger.fps)

        def log_system_usage(self, step, perf):
            """ cpu, gpu, ram usage """
            for n, v in perf.items():
                self._file_writer.add_scalar("sys/"+n, v, global_step=step)

    return FilteredTBXLogger



##########################################################################################
####################################     Tests   ##################################
##########################################################################################


if __name__ == "__main__":
    # test overwrite config 
    pass 