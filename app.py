import os
from flask import Flask
from utils import logger
import config
from auth import start_cleanup_thread
from routes import register_routes

def create_app():
    """创建并配置Flask应用"""
    config.init_config() # 调整到 create_app 开头
    app = Flask(__name__)
    
    # 启动会话清理线程
    start_cleanup_thread()
    
    # 注册路由
    register_routes(app)
    
    return app

if __name__ == "__main__":
    # 初始化配置 # 已移至 create_app
    
    # 创建应用
    app = create_app()
    
    # 获取端口
    port = int(os.getenv("PORT", 7860))
    print(f"[系统] Flask 应用将在 0.0.0.0:{port} 启动 (Flask 开发服)")
    
    # 启动应用
    flask_debug_mode = config.get_config_value("FLASK_DEBUG", default=False) # 从配置获取调试模式
    app.run(host='0.0.0.0', port=port, debug=flask_debug_mode)