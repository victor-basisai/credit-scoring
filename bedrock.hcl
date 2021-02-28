# IMPORTANT: Bedrock HCL version
version = "1.0"

# TODO: name the steps
train {
    step train {
        image = "basisai/workload-standard:v0.2.2"
        install = [
            "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh",
            "bash ~/miniconda.sh -b -p ~/miniconda",
            "rm ~/miniconda.sh",
            "export PATH=$HOME/miniconda/bin:$PATH",
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
        parameters {
            TRAINING_DATA_AWS_BUCKET = "s3://veritas-credit-scoring/data/training/latest"
        }
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
        WORKERS = "1"
    }
}