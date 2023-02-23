FROM gcr.io/deeplearning-platform-release/tf-cpu.2-8
RUN pip install -U fire cloudml-hypertune scikit-learn==0.20.4
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]
