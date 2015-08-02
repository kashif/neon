import logging
from neon.optimizers.learning_rule import LearningRule

logger = logging.getLogger(__name__)


class Adam(LearningRule):

    """
    Adam based learning rule updates. http://arxiv.org/pdf/1412.6980v8.pdf
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        if param_dtype is not None:
            self.param_dtype = param_dtype
        super(Adam, self).__init__(name, lr_params)
        if 'beta_1' in lr_params:
            self.beta_1 = lr_params['beta_1']
        else:
            self.beta_1 = 0.9
        if 'beta_2' in lr_params:
            self.beta_2 = lr_params['beta_2']
        else:
            self.beta_2 = 0.999
        if 'epsilon' in lr_params:
            self.epsilon = lr_params['epsilon']
        else:
            self.epsilon = 1e-8

        self.running_1st_mom_dtype = self.param_dtype
        self.running_2nd_mom_dtype = self.param_dtype
        self.scratch_space_dtype = self.param_dtype
        self.lrates_dtype = self.param_dtype

        self.running_1st_mom = []
        self.running_2nd_mom = []
        self.lrates = []
        self.scratch_space = []
        self.param_names = ['running_1st_mom', 'running_2nd_mom', 'lrates',
                            'scratch_space']

    def allocate_state(self, params):
        assert len(self.running_1st_mom) == 0
        for item in params:
            self.running_1st_mom.append(self.backend.zeros_like(item,
                                   self.running_1st_mom_dtype))
            self.running_2nd_mom.append(self.backend.zeros_like(item,
                                   self.running_2nd_mom_dtype))
            self.lrates.append(self.backend.zeros_like(item,
                               self.lrates_dtype))
            self.scratch_space.append(self.backend.zeros_like(item,
                                      self.scratch_space_dtype))

    def apply_rule(self, params, updates, epoch):
        for ps_item, us_item, ms_item, vs_item, ls_item, ss_item in zip(
                params, updates, self.running_1st_mom,
                self.running_2nd_mom, self.lrates, self.scratch_space):
            self.backend.adam_update(ps_item, us_item, ms_item, vs_item,
                                    ls_item, ss_item, self.beta_1, self.beta_2,
                                    self.epsilon, epoch)
