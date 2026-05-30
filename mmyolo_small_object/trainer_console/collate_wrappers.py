from mmengine.registry import FUNCTIONS


@FUNCTIONS.register_module()
def safe_yolo_collate(batch, **kwargs):
    """包装函数：吸收并丢弃 mmengine 强行注入的 kwargs (如 _scope_, _delete_)。
    PyTorch 传来的 batch 数据被继续传入给对应的真实函数处理并原样挂载。

    使用 lazy import 以避免在非训练环境（仅浏览数据集）中因缺少 mmyolo 而启动失败。
    """
    from mmyolo.datasets.utils import yolov5_collate

    kwargs.pop('_scope_', None)
    kwargs.pop('_delete_', None)
    return yolov5_collate(batch)


@FUNCTIONS.register_module()
def safe_pseudo_collate(batch, **kwargs):
    """Lazy import 版本，避免模块级 import mmengine.dataset。"""
    from mmengine.dataset import pseudo_collate

    kwargs.pop('_scope_', None)
    kwargs.pop('_delete_', None)
    return pseudo_collate(batch)
