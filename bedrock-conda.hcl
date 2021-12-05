# Bedrock HCL schema version (do not change!)
version = "1.0"

# Train Stanza
train {
    # Step
    step train {
        # Baseline Docker image
        image = "continuumio/miniconda3:latest"
        # Install dependencies
        install = [
            "conda env update -f environment-train.yaml",
            "eval \"$(conda shell.bash hook)\"",
            "conda activate veritas"
        ]
        # Entrypoint to main script
        script = [{sh = ["python train.py"]}]
        # Request resources
        resources {
            cpu = "1.0"
            memory = "4G"
        }
        # Only attempt once (default = 3)
        retry {
            limit = 0
        }
    }
    # Environment params shared across all steps in stanza
    parameters {
        DATA_DIR_LOCAL = "data/creditdata"
        SEED = "0"
        CLASSIFIER = "LR"
        TH = "0.5"
        LR_REGULARIZER = "1e-1"
        RF_N_ESTIMATORS = "100"
        CB_ITERATIONS = "100"
    }
}

# Serve Stanza
serve {
    # Baseline Docker image
    image = "python:3.8"
    # Install dependencies
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    # Execute main script
    script = [
        {sh = [
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve:app"
        ]}
    ]
    # Environment params
    parameters {
        # Number of gunicorn workers to use
        WORKERS = "1"
        # Gunicorn log level
        LOG_LEVEL = "INFO"
        # Index page color
        COLOR = "red"
    }
}