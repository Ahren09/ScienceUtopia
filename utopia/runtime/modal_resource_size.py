"""Deploy with modal deploy -m utopia.runtime.modal_resource_size."""

from utopia.runtime.modal_server import RESOURCE_SIZE, deployment, start_server


def serve():
    start_server(RESOURCE_SIZE)


app, serve = deployment(RESOURCE_SIZE, serve)
