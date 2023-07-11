import torch.nn as nn
from torch.optim import AdamW
import robomimic
from robomimic.algo import register_algo_factory_func, PolicyAlgo, RolloutPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel


from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.common.lr_scheduler import get_scheduler

@register_algo_factory_func("diffusion")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the diffusion algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo configimitation

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    cnn_enabled = ("cnn" in algo_config.policy.model.type)
    transformer_enabled = ("transformer" in algo_config.policy.model.type)

    if cnn_enabled:
        algo_class, algo_kwargs = DiffusionAlgoCNN, {}
    elif transformer_enabled:
        algo_class, algo_kwargs = DiffusionAlgoTransformer, {}

    return algo_class, algo_kwargs



class DiffusionAlgoCNN(PolicyAlgo):
    """
    RolloutPolicy (robomimic) to run the diffusion policy with CNN model.
    """

    def _create_networks(self):
        """
        Create networks for diffusion policy.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = DiffusionUnetLowdimPolicy(
            model = ConditionalUnet1D(
                input_dim=self.algo_config.policy.obs_dim
            ),
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.noise_scheduler.num_train_timesteps,
                beta_start=self.algo_config.noise_scheduler.beta_start,
                beta_end=self.algo_config.noise_scheduler.beta_end,
                beta_schedule=self.algo_config.noise_scheduler.beta_schedule,
                variance_type=self.algo_config.noise_scheduler.variance_type,
                clip_sample=self.algo_config.noise_scheduler.clip_sample,
                prediction_type=self.algo_config.noise_scheduler.prediction_type
            ),
            horizon = self.algo_config.policy.horizon,
            obs_dim = self.algo_config.policy.obs_dim,
            action_dim = self.algo_config.policy.action_dim,
            n_action_steps = self.algo_config.policy.n_action_steps,
            n_obs_steps = self.algo_config.policy.n_obs_steps,
            num_inference_steps = self.algo_config.policy.num_inference_steps,
            obs_as_local_cond = self.algo_config.policy.obs_as_local_cond,
            obs_as_global_cond = self.algo_config.policy.obs_as_global_cond,
            pred_action_steps_only = self.algo_config.policy.pred_action_steps_only,
            oa_step_convention = self.algo_config.policy.oa_step_convention
            # kwargs
        ) # TODO verify if this is possible
        self.nets = self.nets.float().to(self.device)
        self._step_count = 0
        
        # configure normalizer
        self.normalizer = LinearNormalizer() # TODO not normalize per batch, but per rollout

        self.nets["policy"].set_normalizer(self.normalizer)


    def _create_optimizer(self):
        # configure training state
        self.optimizer = AdamW(
            params=self.nets.parameters(),
            betas=self.algo_config.optimizer.betas,
            eps=self.algo_config.optimizer.eps,
            lr=self.algo_config.optimizer.lr,
            weight_decay=self.algo_config.optimizer.weight_decay
        )

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            self.algo_config.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.algo_config.training.lr_warmup_steps,
            num_training_steps=(
                self.algo_config.training.batch_size * self.algo_config.training.num_epochs) \
                    // self.algo_config.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self._step_count-1
        )



    def process_batch_for_training(self, batch):
        """
        Process a batch of data to prepare for training.
        
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """

        # TODO remap batch from robomimic to diffusion policy
        # in diffusion policy, batch is yielded by a data loader


        print("batch", batch)

        action = batch["actions"]
        obs = batch["obs"]
        # filter out desired observations
        # obs = {"robot0_eef_pos": obs["robot0_eef_pos"]}
        obs = list(obs.values())
        data = {
            "action": action,
            "obs": obs
        }
        # data = list(data.values())
        # auto normalize
        self.normalizer.fit(data=data, last_n_dims=1)
        # TODO send process to device
        return data
        
    def train_on_batch(self, batch, epoch, validate=False):
        
        # configure ema        
        ema = EMAModel(
            parameters = self.nets["policy"].parameters(),
            inv_gamma = self.algo_config.ema.inv_gamma,
            max_value = self.algo_config.ema.max_value,
            min_value = self.algo_config.ema.min_value,
            power = self.algo_config.ema.power,
            update_after_step = self.algo_config.ema.update_after_step
        )
        
        # compute loss
        raw_loss = self.nets["policy"].compute_loss(batch)
        loss = raw_loss / self.algo_config.training.gradient_accumulate_every
        loss.backward()

        # step optimizer
        if (self._step_count % self.algo_config.training.gradient_accumulate_every) == 0:
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            ema.update(self.nets["policy"])

        self._step_count += 1

        # update ema
        if self.algo_config.training.use_ema:
            ema.step(self.nets["policy"])
        
        # logging
        raw_loss_cpu = raw_loss.item()

        step_log = {
            'train_loss': raw_loss_cpu,
            'global_step': self._step_count,
            'epoch': self.epoch,
            'lr': self.lr_scheduler.get_last_lr()[0]
        }

        return step_log

    def log_info(self, info):
        # TODO return logging dictionry for tensorboard logging
        raise NotImplementedError()
    
    def set_train(self):
        # By default, just calls self.nets.train()
        return super().set_train()
    
    def on_epoch_end(self, epoch):
        # Usually stepping learning rate schedulers (if they are being used)
        return super().on_epoch_end(epoch)
    
    def serialize(self):
        # Returns the state dictionary that contains the current model parameters
        return super().serialize()
    
    ### Testing methods ###

    def set_eval(self):
        # By default, just calls self.nets.eval()
        return super().set_eval()
    
    def deserialize(self, model_dict):
        # Loads model weights - Used at test-time to restore model weights
        return super().deserialize(model_dict)
    
    def get_action(self, obs_dict):
        # Return one or more actions, given observations
        
        return self.nets["policy"].predict_action(obs_dict)
    
    def reset(self):
        # Clear internal agent state before starting a rollout
        return super().reset()
    
    def recreate_model(self, model_dict):
        # Recreates the model from the state dictionary
        self._create_networks()
    
class DiffusionAlgoTransformer(PolicyAlgo):
    """
    RolloutPolicy (robomimic) to run the diffusion policy with Transformer model.
    """
    def __init__(self, model, **kwargs):
        """
        Args:
            model (DiffusionModelTransformer): Transformer model for diffusion policy.
        """
        super().__init__(**kwargs)
        self.model = model

    
    def rollout(self, obs_dict, **kwargs):
        """
        Run the diffusion policy for a single rollout.

        Args:
            obs_dict (dict): dictionary of observations
            **kwargs: additional arguments to pass to policy

        Returns:
            dict: dictionary of rollout results
        """
        return self.model.rollout(obs_dict, **kwargs)