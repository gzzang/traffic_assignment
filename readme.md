# 交通分配问题

+ 给定路段（基于关联矩阵、标记信息）
  + 基于起讫点对路段关联变量
      + 数学规划
      + Frank-Wolfe
        + 数学规划求解下降方向
        + 最短路算法求解下降方向
  + 基于路段变量
    + Frank-Wolfe
      + 数学规划求解下降方向
      + 最短路算法求解下降方向
    + 列生成（其中需要调用基于路径的方法）

+ 给定路径
    + 数学规划
    + 变分不等式
    + MSA

+ preparation.py
  - 文件读取
    + 默认读pkl
    + 没有pkl读cvs并生成pkl
  - 参数设置
  - 数据处理

+ algorithm.py
  - 算法实现

