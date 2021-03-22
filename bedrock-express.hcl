# IMPORTANT: Bedrock HCL version
version = "1.0"

# TODO: name the steps
serve {
    image = "basisai/express-flask:v0.0.3"
    install = [
        "pip3 install --upgrade pip",
        "pip3 install -r requirements-serve.txt",
    ]
    script = [
        {sh = ["python serve-express.py"]}
    ]
    parameters {
        BEDROCK_SERVER = "serve"
    }
}
