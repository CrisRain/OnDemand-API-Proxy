import os
from flask import Flask
from utils import logger, setup_logging
import config
from auth import start_cleanup_thread
from routes import register_routes

def create_app():
    """创建并配置Flask应用"""
    app = Flask(__name__)
    
    # 启动会话清理线程
    start_cleanup_thread()
    
    # 注册路由
    register_routes(app)
    
    return app

if __name__ == "__main__":
    # 初始化配置
    config.init_config()
    
    # 创建应用
    app = create_app()
    
    # 获取端口
    port = int(os.getenv("PORT", 7860))
    print(f"[系统] Flask 应用将在 0.0.0.0:{port} 启动 (使用 Flask 开发服务器)")
    
    # 启动应用
    app.run(host='0.0.0.0', port=port, debug=False) # 或者 debug=True 用于开发调试