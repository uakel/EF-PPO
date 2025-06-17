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
