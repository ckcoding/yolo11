import os
from ultralytics import YOLO

def main():
    onnx_file = "yolo11n.onnx"
    if not os.path.exists(onnx_file):
        print(f"没有检测到文件 {onnx_file} 的存在。系统即将为您联网尝试下载预备权重格式并实施一键转换导出过程...")
        try:
            # 下载原始基准张量权重模型 yolo11n.pt，如果没有也会自动被下载。
            model = YOLO("yolo11n.pt")
            print("模型数据装载成功，开始执行 ONNX 框架转换格式的操作...")
            model.export(format="onnx")
            print("转换和导出流程全部彻底结束。您现在拥有兼容后台加速所必须的要求格式文件了。")
        except Exception as e:
            print(f"由于网络环境问题或者代码不可预期错误导致无法完成导出: {e}")
    else:
        print(f"{onnx_file} 已然处于完备状态，您可以放心启动部署工程代码不需要任何配置了！")

if __name__ == "__main__":
    main()
