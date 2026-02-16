1. Create virtual environment using conda

`conda create -n mlflow-env python=3.9`

2. Activate the Virtual Environment: Before installing MLflow and other dependencies, activate the virtual environment.

`conda activate mlflow-env`

3. Install MLflow: With the virtual environment activated, install MLflow.

`pip install mlflow==2.7.1`

4.  Install Backend Store (Optional)
MLflow uses a tracking server to log experiment data. By default, it logs to the local filesystem, but for more robust use, you may want to set up a database like MySQL or SQLite.

For SQLite (Simpler Option):

SQLite comes pre-installed on many systems, including Ubuntu.
Decide on a directory where you want your SQLite database to reside
```
cd ~/mlflow_server
mkdir metrics_store
```

5. Set Backend Store for MLflow
For SQLite, you'll use a URI like: `sqlite:////home/mlflow/mlflow_server/metrics_store/mlflow.db`

6. Install Artifact Store
The artifact store is where MLflow saves model artifacts like models and plots. You can use S3, Azure Blob Storage, Google Cloud Storage, or even a shared filesystem.

For local storage (simplest for getting started), use a local directory.
```
cd ~/mlflow_server
mkdir artifact_store
```
 
7. Launch MLflow Tracking Server
Open a terminal and run the following command, replacing the URIs with your chosen backend and artifact store paths:

`mlflow server --backend-store-uri sqlite:////home/mlflow/mlflow_server/metrics_store/mlflow.db --default-artifact-root ./artifact_store/mlflow-artifacts`

8. Accessing the MLflow UI
Once the tracking server is running, it will display a URL, typically `http://127.0.0.1:5000`. Open this URL in a web browser to access the MLflow UI.
You can now navigate the UI to see your experiments, runs, metrics, and artifacts.


9. To run some of the experiments, you will need the following
`pip install torch torchvision`
`pip install transformers==4.43.3`