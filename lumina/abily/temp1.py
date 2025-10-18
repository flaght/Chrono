import lightgbm
from lightgbm import libpath

try:
    # libpath._find_lib_path() 是 LightGBM 内部用来寻找 .so 文件的函数
    # 它会返回一个列表，里面是它找到的所有可能的库文件路径
    lib_path_list = libpath._find_lib_path()
    print("LightGBM is trying to load library from one of these paths:")
    for path in lib_path_list:
        print(f"- {path}")
        
    if not lib_path_list:
         print("!!! CRITICAL: LightGBM could not find any lib_lightgbm.so file !!!")

except Exception as e:
    print(f"An error occurred while trying to find the library path: {e}")