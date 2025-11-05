### Summary
This is a project for measuring the size of sweaters. The project is based on the following steps:

## 数据处理
1. 收集数据之后， 使用tools\data_genetation中的程序进行数据处理
2. 处理json标签， 先使用json2xml.py将json标签转换为xml标签
3. 使用cut_out_rag.py 将图像进行分割
4. 使用data_color_aug.py 对图像进行色彩增强
5. 使用xml2xml.py 生成色彩增强图像的标签
6. 使用aug_img.py 对图像进行几何变换，生成多张图像
7. 使用check_xml.py 对生成的标签进行检查， 每个点在图像上是否正确
8. 注意在使用前需要检查label_dict.py中的标签是否对应

## 数据集划分
1.使用tools\data 中的程序进行划分

## 模型训练
1. 在configs中新建配置文件，注意填写数据集路径以及关键点的数量
2. 使用tools\train 中的程序进行模型训练
3. 使用tools\test 中的程序进行模型测试

### Changlogs
    - v1.3.0
        - aug0704.py 可以生成图像并且xml标签是准确的， 但是会有局部位置出现在图像中， 可以考虑另一台电脑中替换背景的方法， 但是换颜色的方法不变
    - v1.2.0
        - 修复数据生成得到图像shape不统一的bug
        - 新增数据生成方式， 可以替换背景， 统一尺寸
        - 修改二阶段解析热力图的逻辑
            ''' 将每一个roi图像进行单独解析，每一个都取top 2， 
                然后将其的分别与中心计算距离， 取距离最近的作为最终的点坐标
            '''
    - v1.1.0
        - 增加数据生成部分， tools\data_genetation