from robomimic.config.base_config import BaseConfig



class DiffusionPolicyConfig(BaseConfig):
    ALGO_NAME = "diffusion"

    def __init__(self, dict_to_load=None):
        super().__init__(dict_to_load=dict_to_load)
         # store algo name class property in the config (must be implemented by subclasses)
        self.algo_name = type(self).ALGO_NAME

        self.experiment_config()
        self.train_config()
        self.algo_config()
        self.observation_config()
        self.meta_config()

        # After Config init, new keys cannot be added to the config, except under nested
        # attributes that have called @do_not_lock_keys
        self.lock_keys()


    def algo_config(self):
        # noise scheduler parameters
        self.algo.noise_scheduler.num_train_timesteps = 100
        self.algo.noise_scheduler.beta_start = 0.0001
        self.algo.noise_scheduler.beta_end = 0.02
        self.algo.noise_scheduler.beta_schedule = "squaredcos_cap_v2"
        self.algo.noise_scheduler.variance_type = "fixed_small" # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
        self.algo.noise_scheduler.clip_sample = True # required when predict_epsilon=False
        self.algo.noise_scheduler.prediction_type = "epsilon" # or sample

        # diffusion policy parameters
        self.algo.policy.action_dim = 7
        self.algo.policy.horizon = 16
        self.algo.policy.n_action_steps = 8
        self.algo.policy.n_obs_steps = 2
        self.algo.policy.num_inference_steps = 100
        self.algo.policy.oa_step_convention = True
        self.algo.policy.obs_as_global_cond = True
        self.algo.policy.obs_as_local_cond = False
        self.algo.policy.obs_dim = 10 + 2
        self.algo.policy.pred_action_steps_only = False
        # Original parameters
        # self.algo.policy.action_dim = 10
        # self.algo.policy.horizon = 16
        # self.algo.policy.n_action_steps = 8
        # self.algo.policy.n_obs_steps = 2
        # self.algo.policy.num_inference_steps = 100
        # self.algo.policy.oa_step_convention = True
        # self.algo.policy.obs_as_global_cond = True
        # self.algo.policy.obs_as_local_cond = False
        # self.algo.policy.obs_dim = 19
        # self.algo.policy.pred_action_steps_only = False

        # diffusion model parameters
        self.algo.policy.model.type = "cnn"
        self.algo.policy.model.cond_predict_scale = True
        self.algo.policy.model.diffusion_step_embed_dim = 256
        self.algo.policy.model.down_dims = (256, 512, 1024)
        self.algo.policy.model.global_cond_dim = 38
        self.algo.policy.model.input_dim = 10
        self.algo.policy.model.kernel_size = 5
        self.algo.policy.model.local_cond_dim = None
        self.algo.policy.model.n_groups = 8

        # ema parameters
        self.algo.ema.inv_gamma = 1.0
        self.algo.ema.max_value = 0.9999
        self.algo.ema.min_value = 0.0
        self.algo.ema.power = 0.75
        self.algo.ema.update_after_step = 0

        # training parameters
        self.algo.training.checkpoint_every = 50
        self.algo.training.debug = False
        self.algo.training.device = "cuda:0"
        self.algo.training.gradient_accumulate_every = 1
        self.algo.training.lr_scheduler = "cosine"
        self.algo.training.lr_warmup_steps = 500
        self.algo.training.max_train_steps = None
        self.algo.training.max_val_steps = None
        self.algo.training.num_epochs = 5000
        self.algo.training.resume = True
        self.algo.training.rollout_every = 50
        self.algo.training.sample_every = 5
        self.algo.training.seed = 42
        self.algo.training.tqdm_interval_sec = 1.0
        self.algo.training.use_ema = True
        self.algo.training.val_every = 1
        self.algo.training.batch_size = 256

        # optimizer parameters
        self.algo.optimizer.betas = (0.95, 0.999)
        self.algo.optimizer.eps = 1.0e-08
        self.algo.optimizer.lr = 0.0001
        self.algo.optimizer.weight_decay = 1.0e-06

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" # learning rate scheduler ("multistep", "linear", etc)
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength
        # TODO remove this