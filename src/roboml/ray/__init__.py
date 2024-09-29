from fastapi import FastAPI
from ray import serve


app = FastAPI(title="roboml HTTP Server")
ingress_decorator = serve.ingress(app)
