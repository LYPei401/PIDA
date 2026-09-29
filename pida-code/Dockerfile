FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /artifact
COPY requirements.txt /artifact/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /artifact
ENV PYTHONPATH=/artifact/atmospheric
CMD ["python", "geology/case_study/scripts/check_artifact.py", "--mode", "source"]
