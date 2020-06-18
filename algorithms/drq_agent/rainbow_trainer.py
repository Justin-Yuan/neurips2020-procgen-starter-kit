rainbow_config = config.copy()
            rainbow_config["num_atoms"] = 10
            rainbow_config["noisy"] = True
            rainbow_config["double_q"] = True
            rainbow_config["dueling"] = True
            rainbow_config["n_step"] = 5
            trainer = dqn.DQNTrainer(config=rainbow_config, env="CartPole-v0")