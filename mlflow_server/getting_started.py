import mlflow
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000") #server url

experiment_id = mlflow.create_experiment("My experiment")

# Run - iterations of experiment
#with mlflow.start_run(experiment_id=experiment_id):
#   pass


run = mlflow.start_run(experiment_id=experiment_id, run_name="First run")

#logging parameters    
mlflow.log_param("learning_rate", 0.01) #configuration setting used for ML models
mlflow.log_param("batch_size", 32)
num_epochs = 10
mlflow.log_param("num_epochs", num_epochs)

#logging metrics - time or ML step based
for epoch in range(num_epochs):
    mlflow.log_metric("accuracy", np.random.random(), step=epoch)
    mlflow.log_metric("loss",np.random.random(), step=epoch)

#logging time series metrics
for t in range(100):
    metric_value = np.sin(t * np.pi / 50)
    mlflow.log_metric("time_series_metric", metric_value, step=t)

'''
# logging artifacts
with open("data/dataset.csv", "w") as f:
    f.write("x,y\n")
    for x in range(100):
        f.write(f"{x},{x+1}\n")

mlflow.log_artifact("data/dataset.csv", "data") #what to log, where it stays on the server relative to the run
'''

#logging models
from transformers import AutoModelForSeq2SeqLM

# Initialize a model from Hugging Face Transformers
model = AutoModelForSeq2SeqLM.from_pretrained("TheFuzzyScientist/T5-base_Amazon-product-reviews")


# Log the model in MLflow
mlflow.pytorch.log_model(model, "transformer_model")

#end run
mlflow.end_run()
