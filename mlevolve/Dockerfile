FROM python:3.11

WORKDIR /app

# 使用清华源加速 pip 安装
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 分开复制和安装每个依赖文件（最大化利用Docker缓存）
COPY requirements_base.txt ./
RUN pip install --no-deps -r requirements_base.txt

COPY requirements_ml.txt ./
RUN pip install --no-deps -r requirements_ml.txt

COPY requirements_domain.txt ./
RUN pip install --no-deps -r requirements_domain.txt

# 复制项目文件
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
