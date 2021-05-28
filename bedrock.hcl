# IMPORTANT: Bedrock HCL version
version = "1.0"

# TODO: name the steps
train {
    step train {
        image = "continuumio/miniconda3:latest"
        install = [
            "conda env update -f environment-train.yaml",
            "eval \"$(conda shell.bash hook)\"",
            "conda activate veritas"
        ]
        script = [{sh = ["python train.py"]}]
        # TODO: Increase the resources... reasonably!
        resources {
            cpu = "1.0"
            memory = "4G"
        }
    }
    parameters {
        DATA_DIR_LOCAL = "data/creditdata"
        SEED = "0"
        TH = "0.5"
        LR_REGULARIZER = "1e-1"
        RF_N_ESTIMATORS = "100"
        CB_ITERATIONS = "100"
    }
}

# TODO: name the steps
serve {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    script = [
        {sh = [
            "gunicorn --bind=:${BEDROCK_SERVER_PORT:-8080} --worker-class=gthread --workers=${WORKERS} --timeout=300 --preload serve:app"
        ]}
    ]
    parameters {
        // Number of gunicorn workers to use
        WORKERS = "1"
        // Gunicorn log level
        LOG_LEVEL = "INFO"
    }
}