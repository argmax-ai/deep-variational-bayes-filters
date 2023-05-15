# Copyright (C) 2019-2023 Volkswagen Aktiengesellschaft,
# Berliner Ring 2, 38440 Wolfsburg, Germany
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional
from typing import Tuple
import numpy as np
import omegaconf
import torch
from torch import distributions as dist
from robolab.robots.robot import Robot
from robolab import utils
from robolab.distributions import MultivariateNormalDiag
from robolab.models.reconstruction import Reconstruction
from robolab.models.latent_state import ProbabilisticLatentState
from robolab.models.networks import Dense
from robolab.models.networks import Transition
from robolab.models.networks import Decoder
from robolab.models.networks import Encoder
from robolab.models.networks import InitialNetwork
from robolab.models.sequence_models.sequence_model import SequenceModel
from robolab.models.trajectory import ProbabilisticTrajectory
from robolab.models.variable import RandomVariable
from .beta_update import BetaUpdate
from .beta_update import ConstrainedOptimization
from .beta_update import BetaAnnealing
from .utils import sensor_fusion


class FusionDVBF(SequenceModel):
    # pylint: disable=too-many-instance-attributes
    # Ignore number of attributes for machine learning models

    _SEQUENCE_MODEL_TYPE = "FusionDVBF"

    def __init__(
        self,
        observation_shape: Tuple[int],
        target_shape: Tuple[int],
        control_shape: Tuple[int],
        n_latent: int,
        initial_net: InitialNetwork,
        initial_transformation,
        encoder: Encoder,
        decoder: Decoder,
        transition: Transition,
        beta_update: BetaUpdate,
    ):  # pylint: disable=too-many-arguments
        """Implementation of the Deep Variational Bayes Filter.

        Karl, Maximilian, et al. "Deep variational bayes filters:
        Unsupervised learning of state space models from raw data."
        arXiv preprint arXiv:1605.06432 (2016).

        Modified version which special fusion inference structure.

        Parameters
        ----------
        observation_shape: Tuple[int]
        target_shape: Tuple[int]
        control_shape: Tuple[int]
        n_latent: int
            Number of latent dimensions.
        initial_net: InitialNetwork
        initial_transformation
        encoder: Encoder
        decoder: Decoder
        transition: Transition
        beta_update: BetaUpdate
        """
        super().__init__()

        self.observation_shape = observation_shape
        self.observation_size = np.prod(self.observation_shape)
        self.target_shape = target_shape
        self.target_size = np.prod(self.target_shape)
        self._control_shape = control_shape
        self.n_latent = n_latent

        self.initial_net = initial_net
        self.initial_transformation = initial_transformation
        self._n_initial_obs = self.initial_net.n_initial_obs

        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition

        self.beta_update = beta_update

    @classmethod
    def from_cfg(cls, cfg: omegaconf.DictConfig, robot: Robot, **kwargs):
        # overwrite n_control
        n_control = kwargs.get("n_control", utils.prod(robot.control_shape))

        initial_network = InitialNetwork.create(cfg.initial_network.type, cfg=cfg, robot=robot)

        encoder = Encoder.create(
            cfg.encoder.type,
            robot=robot,
            input_shape=robot.input_shape,
            n_output=cfg.n_z_latent,
            layers=cfg.encoder.layers,
            units=cfg.encoder.units,
        )

        transition = Transition.create(
            cfg.transition, n_state=cfg.n_z_latent, n_control=n_control, dt=robot.dt
        )

        decoder = Decoder.create(
            cfg.decoder.type,
            robot=robot,
            n_input=cfg.n_z_latent,
            output_shape=robot.target_shape,
            layers=cfg.decoder.layers,
            units=cfg.decoder.units,
        )

        # Setup beta vae update
        if cfg.get("constrained_optimization", False):
            beta_update = ConstrainedOptimization(
                target_size=np.prod(robot.target_shape),
                lambda_dual=cfg.get("lambda_reconstruction", 1.0),
                nu_dual=cfg.get("nu_reconstruction", 1.0),
                eps_dual=cfg.get("eps_reconstruction", 0.5),
            )
        else:
            beta_update = BetaAnnealing(
                beta=cfg.get("beta_z", 1.0),
                temperature=cfg.get("temperature_z", 1.0),
            )

        return cls(
            observation_shape=robot.input_shape,
            target_shape=robot.target_shape,
            control_shape=n_control,
            n_latent=cfg.n_z_latent,
            initial_net=initial_network,
            initial_transformation=Dense(
                cfg.n_z_latent,
                cfg.n_z_latent,
                activation=None,
                hidden_layers=1,
                hidden_units=cfg.initial_network.units,
                hidden_activation=torch.relu,
            ),
            encoder=encoder,
            decoder=decoder,
            transition=transition,
            beta_update=beta_update,
        )

    @property
    def latent_dims(self):
        return (self.n_latent,)

    @property
    def latent_belief_dims(self):
        return (2 * self.n_latent,)

    def forward(
        self, observations: torch.Tensor, controls: torch.Tensor
    ) -> ProbabilisticTrajectory:
        """

        Parameters
        ----------
        observations
            Tensor of observations: (Batch, Timesteps, flattened_observation_shape)
        controls
            Tensor of controls: (Batch, Timesteps, flattened_control_shape)
        """
        latents = self.encode(observations, controls)
        prior = self.prior(latents, controls)
        targets = self._decode(latents)

        return ProbabilisticTrajectory(target=targets, posterior=latents, prior=prior)

    def loss(
        self,
        predictions: ProbabilisticTrajectory,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_step: int = 0,
        **ignored
    ):
        nll_sample_wise = -predictions.target.dist.log_prob(targets)

        if mask is not None:
            nll_sample_wise *= mask

        nll = torch.mean(nll_sample_wise)

        kl_divergences = {}
        posteriors = predictions.posterior.random_variables
        priors = predictions.prior.random_variables
        for i, (posterior, prior) in enumerate(zip(posteriors, priors)):
            kl_sample_wise = dist.kl_divergence(posterior.dist, prior.dist)

            if mask is not None:
                # First timestep has no switching variable, modify mask accordingly
                kl_sample_wise *= mask[-kl_sample_wise.shape[0] :]

            kl = torch.mean(kl_sample_wise)
            kl_divergences[i] = kl

        transition_loss = self.transition.loss()

        loss = self.beta_update.compute_loss(nll, kl_divergences, transition_loss)

        self.beta_update.update_dual_variables(global_step=global_step)

        outputs = {
            "loss": loss,
            "progress_bar": {"sequence_model/loss": loss},
            "log": {
                "sequence_model/loss": loss,
                "sequence_model/nll": nll,
                "sequence_model/kl/z": kl_divergences[0],
                "sequence_model/transition_regularization": transition_loss,
            },
        }

        outputs["log"].update(self.beta_update.log_dict())

        return outputs

    def prefix_predict(self, prefix, controls):
        latents, _ = self.encode(prefix[0], prefix[1])
        predictions = self._generate(latents[0], controls)

        return predictions

    def generate(self, state: ProbabilisticLatentState, controls):
        # generations_initial_latents = self.sample_initial_state_prior(controls.shape[1])
        generations = self._generate(state, controls)
        return generations

    def decode(
        self,
        latent_state: ProbabilisticLatentState,
        control: torch.Tensor,
        deterministic: bool = False,
    ) -> Reconstruction:
        if deterministic:
            return Reconstruction(
                observation=self.decoder(latent_state.state.sample).mean, control=control
            )
        else:
            return Reconstruction(
                observation=self.decoder(latent_state.state.sample).rsample(), control=control
            )

    def initial_state_inference(self, observations, controls) -> ProbabilisticLatentState:
        w1_mean, w1_stddev = self.initial_net(observations, controls)
        w1_dist = MultivariateNormalDiag(w1_mean, w1_stddev)
        w1 = w1_dist.rsample()

        z1 = self.initial_transformation(w1)

        return ProbabilisticLatentState(state=RandomVariable(z1, w1_dist))

    def sample_initial_state_prior(
        self, samples=1, device="cpu", **kwargs
    ) -> ProbabilisticLatentState:
        w1_dist = MultivariateNormalDiag(
            torch.zeros((samples, self.n_latent), device=device),
            torch.ones((samples, self.n_latent), device=device),
        )
        w1 = w1_dist.rsample()

        z1 = self.initial_transformation(w1)

        return ProbabilisticLatentState(state=RandomVariable(z1, w1_dist))

    def encode(self, observations, controls):
        initial_state = self.initial_state_inference(observations, controls)

        inv_meas = self._inverse_measurement_model(observations, controls)
        latents = self._filtered_inference_transition(initial_state, inv_meas, controls[1:])

        return latents

    def prior(self, latents: ProbabilisticLatentState, controls: torch.Tensor):
        mean, _, variance = self.transition(latents.state.sample[:-1], controls[1:])
        mean = torch.cat((torch.zeros_like(mean[0])[None], mean), 0)
        variance = torch.cat((torch.ones_like(variance[0])[None], variance), 0)

        return ProbabilisticLatentState(
            state=RandomVariable(None, MultivariateNormalDiag(mean, torch.sqrt(variance)))
        )

    def _decode(self, latent_state: ProbabilisticLatentState):
        x_reconstructed_dist = self.decoder(latent_state.state.sample)
        x_reconstructed = x_reconstructed_dist.rsample()

        return RandomVariable(x_reconstructed, x_reconstructed_dist)

    def _inverse_measurement_model(self, observations, controls):
        z_enc_mean, z_enc_stddev = self.encoder(observations)

        return {
            "z": {"mean": z_enc_mean, "stddev": z_enc_stddev},
        }

    def _filtered_inference_transition(
        self, initial_state: ProbabilisticLatentState, inv_meas, controls
    ):
        states = [initial_state]

        for i in range(controls.shape[0]):
            state = self._filtered_one_step(
                states[i],
                control=controls[i],
                z_meas_mean=inv_meas["z"]["mean"][i + 1],
                z_meas_stddev=inv_meas["z"]["stddev"][i + 1],
            )

            states.append(state)

        return torch.stack(states)

    def _generate(
        self, initial_state: ProbabilisticLatentState, controls: torch.Tensor
    ) -> ProbabilisticTrajectory:
        states = [initial_state]
        for i in range(controls.shape[0]):
            states.append(self.one_step(states[i], controls[i]))

        gen_states = torch.stack(states)

        reconstructions = self._decode(gen_states)

        return ProbabilisticTrajectory(target=reconstructions, posterior=None, prior=gen_states)

    def _filtered_one_step(
        self, latent_state: ProbabilisticLatentState, control, z_meas_mean, z_meas_stddev
    ) -> ProbabilisticLatentState:
        z_tran_mean, z_tran_var, _ = self.transition(latent_state.state.sample, control)
        z_dist = sensor_fusion(z_tran_mean, z_tran_var, z_meas_mean, z_meas_stddev**2)
        z = z_dist.rsample()

        return ProbabilisticLatentState(RandomVariable(z, z_dist))

    def filtered_one_step(
        self,
        latent_state: ProbabilisticLatentState,
        controls: torch.Tensor,
        observation: torch.Tensor,
    ) -> ProbabilisticLatentState:
        z_enc_mean, z_enc_stddev = self.encoder(observation)

        return self._filtered_one_step(latent_state, controls, z_enc_mean, z_enc_stddev)

    def one_step(
        self, latent_state: ProbabilisticLatentState, controls: torch.Tensor, **kwargs
    ) -> ProbabilisticLatentState:
        z_mean, _, z_var = self.transition(latent_state.state.sample, controls)
        z_dist = MultivariateNormalDiag(z_mean, torch.sqrt(z_var))
        z = z_dist.rsample()

        return ProbabilisticLatentState(RandomVariable(z, z_dist))
