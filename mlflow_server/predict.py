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

# Sample text to predict
texts = [
    "The local high school soccer team triumphed in the state championship, securing victory with a last-second winning goal.",
    "DataCore is set to acquire startup InnovateAI for $2 billion, aiming to enhance its position in the artificial intelligence market.",
]

# Tokenizer needs to be loaded sepparetly for this
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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

print(predict(texts, model, tokenizer))


# Load custom model
model_name = "agnews-transformer"
model_version = "1"  # or "production", "pstaging"
model_version_details = client.get_model_version(name=model_name, version=model_version)

run_id = model_version_details.run_id
artifact_path = model_version_details.source

# Construct the model URI
model_uri = f"models:/{model_name}/{model_version}"

model_path = "models/agnews_transformer"
os.makedirs(model_path, exist_ok=True)

client.download_artifacts(run_id, artifact_path, dst_path=model_path)

# Load the model and tokenizer
custom_model = AutoModelForSequenceClassification.from_pretrained("models/agnews_transformer/custom_model")
tokenizer = AutoTokenizer.from_pretrained("models/agnews_transformer/custom_model")

# Do the inference
print(predict(texts, custom_model, tokenizer))


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


# to clean up
# Delete a specific model version
client.delete_model_version(name=model_name, version=model_version)

# Delete the entire registered model
client.delete_registered_model(name=model_name)
