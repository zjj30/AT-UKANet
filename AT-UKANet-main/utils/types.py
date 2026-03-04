"""
类型转换工具函数
"""
def list_type(s):
    """将逗号分隔的字符串转换为整数列表"""
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

