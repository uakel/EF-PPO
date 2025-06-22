from deprl.vendor.tonic.torch.models import ObservationEncoder
import torch
from ef_ppo.trainers.ef_ppo_trainer import EFPPOTrainer

class ParametricZtrainer(EFPPOTrainer):
    def _test(
        self,
    ):
        if not hasattr(self, "test_fn"):
            return
        score = self.test_fn(
            self.environment,
            self.agent,
        )
        self.max_score = max(getattr(self, "max_score", 0), score)

    def _finish_update(
            self,
            observations,
            muscle_states,
            actions,
            info
    ):
        super()._finish_update(observations, muscle_states, actions, info)
        ends = info["terminations"] | info["resets"]
        if any(ends):
            with torch.no_grad(): 
                tensor_obs = torch.tensor(observations[ends])
                sample = self.agent.model.z_regressor(tensor_obs).sample()
                self._budgets[ends] = sample.cpu().numpy()
