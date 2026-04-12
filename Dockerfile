FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
COPY . .
# 确保这里没有 model.onnx 也可以
EXPOSE 9000
CMD ["python", "app.py"]