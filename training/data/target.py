class SelectTrainingTarget:
    def __init__(self, target_type=None):
        self.target_type = target_type

    def __call__(self, sample):
        return sample
