from ef_ppo.trainers.parametric_z_trainer import ParametricZtrainer

def z_regression_trainer_maker():
    l_buffer = 256
    n_updates_per_epoch = 50
    n_parallel = 20
    n_sequential = 8

    epoch_steps = l_buffer * n_parallel * n_sequential * n_updates_per_epoch
    n_epochs = 150

    trainer = ParametricZtrainer(
        steps=n_epochs * epoch_steps,
        epoch_steps=epoch_steps,
        save_steps=epoch_steps,
        use_env_constraint=True,
        max_budget=0,
        min_budget=-2,
        constraint_type="max",
        budget_update="paper",
        test_fn="ef_ppo.z_regressor_training:train",
    )

    return trainer
