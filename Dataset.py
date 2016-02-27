class DataSet():

    def __init__(self, data, r_seed=None, filter_fn=None):
        self.data = filter(filter_fn, data)
        if r_seed is not None:
            import random
            random.seed(r_seed)
            random.shuffle(self.data)


    def generate_train_test(self, percent, filter_function=None):
        divider = int(len(self.data) * percent)

        train = []
        for i in range(0, divider):
            train.append(self.data[i])

        test = []
        for i in range(divider, len(self.data)):
            test.append(self.data[i])

        return train, test


    def generate_n_cross_validation_sets(self, n):
        if n <= 1:
            raise ValueError('must have n > 1')

        import math

        validation_set_size = int(math.floor(len(self.data) / n))

        sets = []

        for fold in range(0, n):
            training_set = self.data[:fold * validation_set_size] + self.data[(fold + 1) * validation_set_size:]
            validation_set = self.data[fold * validation_set_size:(fold + 1) * validation_set_size]

            sets.append((training_set, validation_set))

        return sets