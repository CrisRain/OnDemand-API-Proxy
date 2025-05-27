import json
import os
from logging_config import logger
from typing import Dict, Any, Optional

def load_config(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置信息。
    
    加载顺序:
    1. 如果提供了 file_path 参数，则尝试从该路径加载。
    2. 如果未提供 file_path，检查 APP_CONFIG_PATH 环境变量，如果设置了，则尝试从该路径加载。
    3. 如果以上都没有提供，则尝试从相对于此文件目录的 config.json 加载。

    Args:
        file_path (Optional[str]): 要加载的配置文件的可选路径。

    Returns:
        Dict[str, Any]: 加载的配置字典，如果加载失败或未找到文件则为空字典。
    """
    config_file_to_load = None
    if file_path:
        config_file_to_load = file_path
        logger.info(f"尝试从参数指定的路径加载配置: {config_file_to_load}")
    else:
        env_path = os.environ.get("APP_CONFIG_PATH")
        if env_path:
            config_file_to_load = env_path
            logger.info(f"尝试从环境变量 APP_CONFIG_PATH 加载配置: {config_file_to_load}")
        else:
            # 使用默认路径 'config.json' 相对于当前目录
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
            config_file_to_load = default_config_path
            logger.info(f"尝试从默认路径加载配置: {config_file_to_load}")

    config = {}
    if config_file_to_load and os.path.exists(config_file_to_load):
        try:
            with open(config_file_to_load, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"已从 {config_file_to_load} 加载配置")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"从 {config_file_to_load} 加载配置文件失败: {e}")
            config = {} # 加载失败视为空配置
    else:
         logger.warning(f"配置文件未找到或路径无效: {config_file_to_load}")
         config = {} # 未找到文件视为空配置

    return config