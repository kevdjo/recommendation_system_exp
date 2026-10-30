"""MLflow / UC Model Registry helpers."""


def registered_model_name(catalog: str, schema: str, model_name: str) -> str:
    return f"{catalog}.{schema}.{model_name}"


def model_uri(catalog: str, schema: str, model_name: str, alias: str) -> str:
    return f"models:/{registered_model_name(catalog, schema, model_name)}@{alias}"
