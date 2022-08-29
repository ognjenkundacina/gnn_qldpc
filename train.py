import ignnition

import predict


#def train():
#    model = ignnition.create_model(model_dir='./')
#    model.computational_graph()
#    model.train_and_validate()


#train()

model = ignnition.create_model(model_dir='./')
model.computational_graph()
model.train_and_validate()

predict(model)
