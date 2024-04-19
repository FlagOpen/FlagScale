# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate decay and weight decay incr functions."""

import math

from .utils import print_rank_0

class OptimizerParamScheduler(object):
    """Anneals learning rate and weight decay"""

    def __init__(self, optimizer, init_lr, max_lr, min_lr,
                 lr_warmup_steps, lr_decay_steps, lr_decay_style,
                 start_wd, end_wd, wd_incr_steps, wd_incr_style,
                 use_checkpoint_opt_param_scheduler=True,
                 override_opt_param_scheduler=False,
                 stablelm2_scheduler_config=None):

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, 'both override and '\
                'use-checkpoint are set.'

        self.stablelm2_scheduler_config = stablelm2_scheduler_config
        if self.stablelm2_scheduler_config is not None:
          ## absolute samples
          self.stablelm2_scheduler_config.rsqrt_samples += \
              self.stablelm2_scheduler_config.cosine_samples
          ## N of consine
          if self.stablelm2_scheduler_config.cosine_period_samples == 0:
            self.stablelm2_scheduler_config.cosine_period_samples = self.lr_decay_steps

        # Set the learning rate
        self.step(0)
        print_rank_0('> learning rate decay style: {}'.format(self.lr_decay_style))


    def get_wd(self):
        """ Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd

        if self.wd_incr_style == 'constant':
            assert self.start_wd == self.end_wd
            return self.end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception('{} weight decay increment style is not supported.'.format(
                self.wd_incr_style))

        return self.start_wd + coeff * delta_wd


    def get_lr(self, param_group):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        max_lr = param_group.get('max_lr', self.max_lr)
        min_lr = param_group.get('min_lr', self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return (
                self.init_lr
                + (
                    (max_lr - self.init_lr)
                    * float(self.num_steps)
                    / float(self.lr_warmup_steps)
                )
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # stablelm2 scheduler of multiple stages
        print_rank_0('> stablelm2_scheduler_config: {}'.format(self.stablelm2_scheduler_config))
        if self.stablelm2_scheduler_config is not None:
          if self.num_steps <= self.stablelm2_scheduler_config.cosine_samples:
              ## cosine phase
              # decay_ratio = float(self.num_steps) / float(self.lr_decay_steps)
              # TODO
              decay_ratio = float(self.num_steps) / float(self.stablelm2_scheduler_config.cosine_period_samples)
              cosine_min_lr = self.stablelm2_scheduler_config.cosine_max_lr * 0.1
              delta_lr = self.stablelm2_scheduler_config.cosine_max_lr - cosine_min_lr
              coeff = 0.5 * (math.cos(2 * math.pi * decay_ratio) + 1.0)
              self.stablelm2_scheduler_config.cosine_lr = cosine_min_lr + coeff * delta_lr
              return self.stablelm2_scheduler_config.cosine_lr
          elif self.num_steps <= self.stablelm2_scheduler_config.rsqrt_samples:
              ## rsqrt phase
              alpha = self.stablelm2_scheduler_config.alpha
              beta = self.stablelm2_scheduler_config.beta
              gbs = self.stablelm2_scheduler_config.global_batch_size * 1.0
              self.stablelm2_scheduler_config.rsqrt_lr = alpha / ((self.num_steps / gbs + beta) ** 0.5)
              return self.stablelm2_scheduler_config.rsqrt_lr
          elif self.stablelm2_scheduler_config.decay_samples <= 0:
              ## optional linear phase
              decay_steps_ = self.lr_decay_steps - self.stablelm2_scheduler_config.rsqrt_samples
              num_steps_ = self.num_steps - self.stablelm2_scheduler_config.rsqrt_samples
              decay_ratio = float(num_steps_) / float(decay_steps_)
              coeff = (1.0 - decay_ratio)
              return coeff * self.stablelm2_scheduler_config.rsqrt_lr
          else:
              ## optional linear phase
              valid_lr_decay_steps_ = min(
                  self.lr_decay_steps,
                  self.stablelm2_scheduler_config.rsqrt_samples + self.stablelm2_scheduler_config.decay_samples)
              if self.num_steps <= valid_lr_decay_steps_:
                decay_steps_ = valid_lr_decay_steps_ - self.stablelm2_scheduler_config.rsqrt_samples
                num_steps_ = self.num_steps - self.stablelm2_scheduler_config.rsqrt_samples
                decay_ratio = float(num_steps_) / float(decay_steps_)
                coeff = (1.0 - decay_ratio)
                delta_lr = self.stablelm2_scheduler_config.rsqrt_lr - self.min_lr
                assert decay_ratio >= 0.0
                return coeff * delta_lr + self.min_lr
              else:
                return self.min_lr

        # Warmup-Stable-Decay(WSD)
        if self.lr_decay_style == 'warmup-stable-decay':
            W = self.lr_warmup_steps
            S = round((self.lr_decay_steps - W) * 10. / 11.)
            ## D is 10% of S.
            T = self.lr_decay_steps - W - S
            ## Warmup Phase, see above
            ## Stable Phase
            if self.num_steps < S:
                return self.max_lr
            else: # Decay Phase
                return self.max_lr * 0.5 ** ((self.num_steps - S) / T)

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps ** 0.5 / (num_steps ** 0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.lr_decay_style))

        return min_lr + coeff * delta_lr


    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            new_lr = self.get_lr(param_group)
            param_group['lr'] = new_lr * param_group.get('lr_mult', 1.0)
            param_group['weight_decay'] = new_wd * param_group.get('wd_mult', 1.0)


    def state_dict(self):
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': self.lr_decay_style,
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': self.wd_incr_style,
            'wd_incr_steps': self.wd_incr_steps
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            print_rank_0(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, \
                f'OptimizerParamScheduler: class input value {cls_value} and checkpoint' \
                f'value {sd_value} for {name} do not match'
        print_rank_0(' > using checkpoint value {} for {}'.format(sd_value,
                                                                  name))
        return sd_value


    def load_state_dict(self, sd):

        if 'start_lr' in sd:
            max_lr_ = sd['start_lr']
        else:
            max_lr_ = sd['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')

        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')

        if 'warmup_iter' in sd:
            lr_warmup_steps_ = sd['warmup_iter']
        elif 'warmup_steps' in sd:
            lr_warmup_steps_ = sd['warmup_steps']
        else:
            lr_warmup_steps_ = sd['lr_warmup_steps']
        self.lr_warmup_steps = self._check_and_set(self.lr_warmup_steps,
                                                lr_warmup_steps_,
                                                'warmup iterations')

        if 'end_iter' in sd:
            lr_decay_steps_ = sd['end_iter']
        elif 'decay_steps' in sd:
            lr_decay_steps_  = sd['decay_steps']
        else:
            lr_decay_steps_ = sd['lr_decay_steps']
        self.lr_decay_steps = self._check_and_set(self.lr_decay_steps, lr_decay_steps_,
                                               'total number of iterations')

        if 'decay_style' in sd:
            lr_decay_style_ = sd['decay_style']
        else:
            lr_decay_style_ = sd['lr_decay_style']
        self.lr_decay_style = self._check_and_set(self.lr_decay_style,
                                               lr_decay_style_,
                                               'learning rate decay style')

        if 'num_iters' in sd:
            num_steps = sd['num_iters']
        else:
            num_steps = sd['num_steps']
        self.step(increment=num_steps)


        if 'start_wd' in sd:
            self.start_wd = self._check_and_set(self.start_wd,
                                                sd['start_wd'],
                                                "start weight decay")
            self.end_wd = self._check_and_set(self.end_wd,
                                                sd['end_wd'],
                                                "end weight decay")
            self.wd_incr_steps = self._check_and_set(self.wd_incr_steps,
                                                sd['wd_incr_steps'],
                                                "total number of weight decay iterations")
            self.wd_incr_style = self._check_and_set(self.wd_incr_style,
                                                sd['wd_incr_style'],
                                                "weight decay incr style")