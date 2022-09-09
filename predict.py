import ignnition
from predict_helper import predict

model = ignnition.create_model(model_dir='./')

predict(model)

