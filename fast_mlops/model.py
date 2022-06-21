

class ModelWrapper:
    def __init__(self, model):
        self.model = model


    def train(self, datas, labels):
        print('>> MODEL TRAIN START ..')
        self.model.fit(datas, labels)