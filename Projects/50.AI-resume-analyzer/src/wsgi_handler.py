# src/wsgi_handler.py

import awsgi
from .app import app # Import ứng dụng Flask của bạn từ package

def lambda_handler(event, context):
    """
    Hàm này là điểm vào cho AWS Lambda.
    Nó chuyển đổi sự kiện Lambda thành một yêu cầu WSGI
    và chuyển nó đến ứng dụng Flask.
    """
    return awsgi.response(app, event, context)