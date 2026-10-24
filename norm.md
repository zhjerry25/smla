## 仓库规范

**远程-本地同步**

代码文件 Git+Git Remote

数据文件 本地下载+rsync

**目标环境**

Nvidia RTX 5090

无多卡并行，有多卡独立运行

**目标模型**

标准 Pre-Norm RoPE **MLA** dense Transformer 实现

**目标尺寸**

约 1B 为核心，附 125M / 250M / 500M

**目标数据集**

FineWeb-Edu 子集

---

**核心原则：最佳性能**

---

**测试条件**

本机 CPU 仅检验

实际 MFU 优化需进行 **修改-测试-反馈-修改** 循环
