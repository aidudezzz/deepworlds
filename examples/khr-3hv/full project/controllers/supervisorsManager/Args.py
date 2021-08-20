class ModelArgs(object):
    """
        Class having all the hyperparameters of the RL model
    """
    def __init__(self, lr, gamma, gae_lambda, clip_param, kl_coeff, num_sgd_iter,
                sgd_minibatch_size, train_batch_size, num_workers, num_gpus, batch_mode, 
                observation_filter):

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.kl_coeff = kl_coeff
        self.num_sgd_iter = num_sgd_iter
        self.sgd_minibatch_size = sgd_minibatch_size
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.batch_mode = batch_mode
        self.observation_filter = observation_filter