# Use official Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 复制核心应用文件
COPY /* .

RUN chmod -R 0755 /app

# Expose the port (Flask 默认端口)
EXPOSE 5000

# 设置 UTF-8 避免中文乱码
ENV LANG=C.UTF-8

# 启动主程序
CMD ["python", "app.py"]
