蟒蛇实验 / Python Experiments
AI 实训项目合集，包含图像分类、表情识别等多个实践项目。
数据集
所有项目使用的数据集已上传至 Hugging Face Datasets：
| 数据集 | 说明 | 链接 |
|--------|------|------|
| 蔬果病害识别数据集 | 蔬果病害图像分类 | [nzg-practice-project-dataset](https://huggingface.co/datasets/NZG556/nzg-practice-project-dataset) |
| Fruits-360 | 水果蔬菜图像分类 | [nzg-practice-project-dataset](https://huggingface.co/datasets/NZG556/nzg-practice-project-dataset) |
| FER2013 | 人脸表情识别 | [nzg-practice-project-dataset](https://huggingface.co/datasets/NZG556/nzg-practice-project-dataset) |
下载方式
```python
from datasets import load_dataset
# 示例：加载数据集
ds = load_dataset("NZG556/nzg-practice-project-dataset")
