"""
分类模型库
This is the libray of classification model.
"""

# 版本信息
__version__ = '0.1.0'

# 作者信息
__author__ = 'Chriss Zhang'

# 可以在这里导入子模块，使它们在包级别可用
from .model import *

# 可以定义 __all__ 来控制使用 from package import * 时导入的内容
__all__ = ['config', 'data', 'train', 'util']