import os 
import sys 
import math 
import logging 
import numpy as np 

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


class NoisyLinear(nn.Linear):
    """ Noisy linear layer with independent Gaussian noise (for Rainbow)
    reference: https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py#L10-L37
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            nn.init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            nn.init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            nn.init.constant(self.sigma_weight, self.sigma_init)
            nn.init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        # set noise first, remember to set model flag train() or eval()
        if self.training:
            self.sample_noise()
        else:
            self.remove_noise()
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)






##########################################################################################
####################################     Torch func Utils   ##################################
#########################################################################################


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


def update_params(src_params, target_params, tau=1.0):
    """ inputs are params directly
    reference: https://github.com/MishaLaskin/curl/blob/019a229eb049b9400e97f142f32dd47b4567ba8a/utils.py#L123
    """
    if tau != 1.0:  # soft update 
        for param, target_param in zip(src_params, target_params):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data)   
    else:   # hard update 
        for param, target_param in zip(src_params, target_params):
            target_param.data.copy_(param.data) 


def update_model_params(model, target_model, tau=1.0):
    """ general hard/soft update func, inputs are models 
    """
    # # Version 1
    # # reference: https://github.com/MishaLaskin/curl/blob/019a229eb049b9400e97f142f32dd47b4567ba8a/utils.py#L123
    # if tau != 1.0:  # soft update 
    #     for param, target_param in zip(model.parameters(), target_model.parameters()):
    #         target_param.data.copy_(
    #             tau * param.data + (1 - tau) * target_param.data)   
    # else:   # hard update 
    #     for param, target_param in zip(model.parameters(), target_model.parameters()):
    #         target_param.data.copy_(param.data)  

    # Version 2 
    # reference: https://github.com/ray-project/ray/blob/master/rllib/agents/sac/sac_torch_policy.py
    model_state_dict = model.state_dict()
    # If tau == 1.0: Full sync from Q-model to target Q-model.
    if tau != 1.0:
        target_state_dict = target_model.state_dict()
        model_state_dict = {
            k: tau * model_state_dict[k] + (1 - tau) * v
            for k, v in target_state_dict.items()
        }
    target_model.load_state_dict(model_state_dict)


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