// Bedrock HCL schema version (do not change!)
version = "1.0"

serve {
    // Bedrock Express image
    image = "basisai/express-flask:v0.0.3"
    // Installing dependencies
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    // Special entrypoint for Bedrock Express
    script = [
        {sh = [
            "/app/entrypoint.sh"
        ]}
    ]
    parameters {
        // This should be the name of python module that has a subclass of BaseModel 
        // https://github.com/basisai/bedrock-express#creating-a-model-server
        // If not specified as a parameter it defaults to "serve"
        BEDROCK_SERVER = "serve-express"
        // Number of gunicorn workers to use
        WORKERS = "1"
        // Gunicorn log level
        LOG_LEVEL = "INFO"
    }
}
