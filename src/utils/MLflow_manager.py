import mlflow
from mlflow.tracking import MlflowClient

def register_and_promote_model(run_id: str, model_name: str, stage: str = 'staging'):
    '''
        Regisiter a model in the MLflow Model Registry and promote it to stage
    '''
    client = MlflowClient()

    # register the model
    result = client.create_model_version(
        name = model_name,
        source = f'{mlflow.get_tracking_uri()}/artifacts/{model_name}',
        run_id=run_id
    )

    version = result.version
    print(f'Registered {model_name} version {version} in MLflow')

    # transition the model to a specified stage
    client.transition_model_version_stage(
        name = model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )

    print(f'Promoted {model_name} version {version} to {stage} stage')
