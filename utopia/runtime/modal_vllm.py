"""Deploy with modal deploy -m utopia.runtime.modal_vllm."""

from utopia.runtime.modal_server import STANDARD, deployment, start_server


def serve():
    start_server(STANDARD)


app, serve = deployment(STANDARD, serve)
