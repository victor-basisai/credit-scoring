# Bedrock HCL schema version (do not change!)
version = "1.0"

serve {
    # Docker image for Bedrock Express
    image = "basisai/express-flask:v0.0.3"
    # Install dependencies
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    # Special Entrypoint for Bedrock Express (do not change!)
    script = [
        {sh = [
            "/app/entrypoint.sh"
        ]}
    ]
    parameters {
        # Special parameter: the name of python module that has a subclass of BaseModel 
        # If not specified as a parameter it defaults to "serve"
        BEDROCK_SERVER = "serve-express"
        # Number of gunicorn workers to use
        WORKERS = "1"
        # Gunicorn log level
        LOG_LEVEL = "INFO"
    }
}
