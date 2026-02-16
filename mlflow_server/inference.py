import mlflow
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

# Load a specific model version
model_name = "agnews_pt_classifier"
model_version = "1"  # or "production", "staging"


model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pytorch.load_model(model_uri)


def predict(texts, model, tokenizer):
    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

    # Pass the inputs to the model
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert predictions to text labels
    predictions = predictions.cpu().numpy()
    predictions = [model.config.id2label[prediction] for prediction in predictions]

    # Print predictions
    return predictions


# Sample text to predict
texts = [
    "The local high school soccer team triumphed in the state championship, securing victory with a last-second winning goal.",
    "DataCore is set to acquire startup InnovateAI for $2 billion, aiming to enhance its position in the artificial intelligence market.",
]

# Tokenizer needs to be loaded sepparetly for this
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print(predict(texts, model, tokenizer))


# Log some new models for versioning demonstration
mlflow.set_experiment("sequence_classification")

# Log a new model as iteration 2
with mlflow.start_run(run_name="iteration2"):
    mlflow.pytorch.log_model(model, "model")

# Log another new model as iteration 3
with mlflow.start_run(run_name="iteration3"):
    mlflow.pytorch.log_model(model, "model")


# Model version management
model_versions = client.search_model_versions(f"name='{model_name}'")
for version in model_versions:
    print(f"Version: {version.version}, Stage: {version.current_stage}")

# Change model stage
client.transition_model_version_stage(name=model_name, version=model_version, stage="Production")


'''
# Delete a specific model version
client.delete_model_version(name=model_name, version=model_version)

# Delete the entire registered model
client.delete_registered_model(name=model_name)
'''