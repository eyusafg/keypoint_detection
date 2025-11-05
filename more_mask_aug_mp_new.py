import os
import cv2
import random
import numpy as np
import albumentations as A
from lxml import etree
import time
from tqdm import tqdm
import multiprocessing as mp
import glob
import xml.etree.ElementTree as ET

# 全局颜色列表
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]

def generate_colors(num_colors):
    """生成指定数量的随机颜色"""
    colors = []
    for _ in range(num_colors):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

def apply_color_transform(image, mask, target_color):
    """对图像应用颜色变换"""
    # 转换图像到LAB色彩空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 创建掩码区域的蒙版
    mask_bool = mask > 0
    
    # 计算掩码区域的平均LAB值
    mean_l = np.mean(lab_image[mask_bool, 0])
    mean_a = np.mean(lab_image[mask_bool, 1])
    mean_b = np.mean(lab_image[mask_bool, 2])
    
    # 将目标颜色转换为LAB色彩空间
    target_lab = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2LAB)[0][0]
    
    # 计算颜色差异
    delta_l = target_lab[0] - mean_l
    delta_a = target_lab[1] - mean_a
    delta_b = target_lab[2] - mean_b
    
    # 增强颜色变化强度
    color_enhancement_factor = 1.5  # 颜色增强因子
    delta_l *= color_enhancement_factor
    delta_a *= color_enhancement_factor
    delta_b *= color_enhancement_factor
    
    # 在掩码区域内应用相对颜色变换
    # 保持原始图像的纹理细节，但改变颜色
    lab_image[mask_bool, 0] = np.clip(lab_image[mask_bool, 0] + delta_l, 0, 255)
    lab_image[mask_bool, 1] = np.clip(lab_image[mask_bool, 1] + delta_a, 0, 255)
    lab_image[mask_bool, 2] = np.clip(lab_image[mask_bool, 2] + delta_b, 0, 255)
    
    # 转换回BGR色彩空间
    result_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    
    # 创建掩码的三通道版本
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 使用掩码合并原图和变换后的图像
    # 调整纹理保留比例以获得更明显的颜色变化效果
    texture_preservation = 0.6  # 调整纹理保留比例
    result_image = np.where(mask_3channel > 0, 
                           cv2.addWeighted(image, texture_preservation, result_image, 1 - texture_preservation, 0), 
                           image)
    if result_image.dtype != np.uint8:
        result_image = np.clip(result_image, 0, 255).astype(np.uint8)    
    return result_image

def generate_variants(img_path, mask_path, mask_path_):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_path_ is not None:
        mask_ = cv2.imread(mask_path_, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape[:2]

    # 计算物体的边界框和中心点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None
    x, y, w, h = cv2.boundingRect(contours[0])
    center_x, center_y = x + w//2, y + h//2
    
    # 计算安全padding
    max_dim = max(w, h)
    padding = int(max_dim * 0.3)  # 添加30%的padding作为安全区域
    
    transform_list = [
        # 步骤1: 安全padding
        # A.PadIfNeeded(
        #     min_height=H + padding*2,
        #     min_width=W + padding*2,
        # border_mode=cv2.BORDER_CONSTANT,  # 使用恒定值填充
        # value=0,  # 填充黑色背景
        #     p=1.0
        # ),
        
        # # 步骤2: 仅使用安全的空间变换
        # A.OneOf([
        #     # 小角度旋转
        #     A.Rotate(
        #         limit=20,  # 限制旋转角度在±20度
        #         border_mode=cv2.BORDER_CONSTANT,
        #         p=1.0
        #     ),
        #     # 轻微的缩放
        #     A.RandomScale(
        #         scale_limit=(0.8,1.2),
        #         interpolation=cv2.INTER_LINEAR,
        #         p=1.0
        #     ),
        # ], p=0.7),

        # A.SafeRotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),

        # 步骤3: 简单的翻转
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        
        # 步骤4: 增强的颜色变换（不影响形状）
        A.OneOf([
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.3,
            #     contrast_limit=0.3,
            #     p=1.0
            # ),
            A.HueSaturationValue(
                hue_shift_limit=180,
                sat_shift_limit=180,
                val_shift_limit=0,
                p=0.5
            ),
            A.RGBShift(
                r_shift_limit=60,
                g_shift_limit=60,
                b_shift_limit=60,
                p=0.5
            ),
        ], p=1.0),
        
        # 最后: 中心裁剪回原始大小
        A.CenterCrop(height=H, width=W, p=1.0),
    ]

    transform_with_mask = A.Compose(transform_list, additional_targets={'mask_': 'mask'})
    transform_without_mask = A.Compose(transform_list, additional_targets=None)

    if mask_path_ is not None:
        augmented = transform_with_mask(image=image, mask=mask, mask_=mask_)
    else:
        augmented = transform_without_mask(image=image, mask=mask)
    
    # 严格的质量检查
    aug_mask = augmented['mask']
    if mask_path_ is not None:
        aug_mask_ = augmented['mask_']

    original_pixels = np.count_nonzero(mask)
    augmented_pixels = np.count_nonzero(aug_mask)

    # 确保增强后的mask面积变化不超过10%
    area_ratio = augmented_pixels / original_pixels
    if 0.9 <= area_ratio <= 1.1:
        # 检查mask的连通性
        num_labels, _ = cv2.connectedComponents(aug_mask)
        if num_labels == 2:  # 背景算一个连通区域，所以总数应该是2
            if mask_path_ is not None:
                return augmented['image'], aug_mask, aug_mask_
            else:
                return augmented['image'], aug_mask, None
            
    return None, None, None

def save_xml_segment(mask, mask_, img_name, xml_temp_dir):
    """保存XML片段到临时文件"""
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    if mask_ is not None:
        contour_ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)

    h, w = mask.shape
    image_id = int(time.time() * 1000) + random.randint(1000, 9999)  # 更唯一的时间戳
    
    # 使用标准的xml.etree.ElementTree，它支持pickle
    img_child = ET.Element('image')
    img_child.set('id', str(image_id))
    img_child.set('name', '%s' % img_name)
    img_child.set('subset', 'Train')
    img_child.set('width', str(w))
    img_child.set('height', str(h))

    polygon_child = ET.Element('polygon')
    polygon_points = []
    for pt in contour:
        polygon_points.append('%.2f,%.2f' % (pt[0], pt[1]))

    polygon_child.set('label', '衣服')
    polygon_child.set('occlued', str(0))
    polygon_child.set('source', 'augmented_with_trans')
    polygon_child.set('z_order', '0')
    polygon_child.set('points', ';'.join(polygon_points))
    img_child.append(polygon_child)

    # if mask_ is not None:
    #     polygon_child1 = ET.Element('polygon')
    #     polygon_points1 = []
    #     for pt in contour_:
    #         polygon_points1.append('%.2f,%.2f' % (pt[0], pt[1]))

    #     polygon_child1.set('label', 'mask')
    #     polygon_child1.set('occlued', str(0))
    #     polygon_child1.set('source', 'augmented_with_trans')
    #     polygon_child1.set('z_order', '0')
    #     polygon_child1.set('points', ';'.join(polygon_points1))
    #     img_child.append(polygon_child1)

    # 将XML元素保存到临时文件
    temp_xml_path = os.path.join(xml_temp_dir, f"{img_name.split('.')[0]}_{image_id}.xml")
    tree = ET.ElementTree(img_child)
    tree.write(temp_xml_path, encoding='utf-8', xml_declaration=True)
    
    return temp_xml_path

def process_single_image(args):
    """处理单个图像的多进程函数"""
    img_name, img_path, mask_path, mask_path_, save_img_path, xml_temp_dir, task_id = args
    
    try:
        img_path_full = os.path.join(img_path, img_name)
        mask_path1 = os.path.join(mask_path, img_name)
        mask_path_1 = os.path.join(mask_path_, img_name) if mask_path_ else None
        
        if not os.path.exists(mask_path_1):
            mask_path_1 = None

        augmented_imgs, augmented_masks, augmented_masks_ = generate_variants(img_path_full, mask_path1, mask_path_1)
        # cv2.namedWindow('augmented_masks', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('augmented_masks_', cv2.WINDOW_NORMAL)
        # cv2.imshow('augmented_masks', augmented_masks)
        # cv2.imshow('augmented_masks_', augmented_masks_)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if augmented_imgs is None:
            return None

        color = random.choice(COLORS)
        if mask_path_1 is None:
            result_img = apply_color_transform(augmented_imgs, augmented_masks, color)
        else:
            result_img = apply_color_transform(augmented_imgs, augmented_masks_, color)

        # 生成新的图像名称
        img_name_new = img_name.split('.')[0] + f'_{task_id}.png'
        
        # 保存图像
        cv2.imwrite(os.path.join(save_img_path, img_name_new), result_img)
        
        # 保存XML片段
        if mask_path_1 is None:
            xml_path = save_xml_segment(augmented_masks, None, img_name_new, xml_temp_dir)
        else:
            xml_path = save_xml_segment(augmented_masks, augmented_masks_, img_name_new, xml_temp_dir)
        
        return img_name_new, xml_path
    except Exception as e:
        print(f"处理图像 {img_name} 时出错: {e}")
        return None

def merge_xml_files(xml_temp_dir, output_xml_path):
    """合并所有XML片段文件"""
    # 创建根元素
    root = ET.Element('annotations')
    
    # 获取所有临时XML文件
    xml_files = glob.glob(os.path.join(xml_temp_dir, "*.xml"))
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            xml_root = tree.getroot()
            # 将每个image元素添加到根元素中
            root.append(xml_root)
        except Exception as e:
            print(f"解析XML文件 {xml_file} 时出错: {e}")

    # 美化XML输出
    def prettify(element, indent='  '):
        """美化XML元素，添加缩进和换行"""
        queue = [(0, element)]  # (level, element)
        while queue:
            level, element = queue.pop(0)
            children = [(level + 1, child) for child in list(element)]
            if children:
                element.text = '\n' + indent * (level + 1)  # 为有子元素的元素添加换行和缩进
            if queue:
                element.tail = '\n' + indent * queue[0][0]  # 为元素尾部添加换行和缩进
            else:
                element.tail = '\n' + indent * (level - 1)  # 最后一个元素
            queue[0:0] = children  # 将子元素添加到队列前面

    prettify(root)     

    # 保存合并后的XML文件
    tree = ET.ElementTree(root)
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    
    # 清理临时文件
    for xml_file in xml_files:
        try:
            os.remove(xml_file)
        except:
            pass

def main():
    img_path = r'C:\thor_datasets\thor_1030\images'
    mask_path = r'C:\thor_datasets\cutout\thor_1030\mask\masks_cloth'
    mask_path_ = r'C:\thor_datasets\cutout\thor_1030\mask\masks'

    save_img_path = 'Dataset/gending/thor_1030/imgs'
    save_xml_path = 'Dataset/gending/thor_1030'
    xml_temp_dir = 'Dataset/gending/thor_1030/temp_xml'
    
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    if not os.path.exists(save_xml_path):
        os.makedirs(save_xml_path)
    if not os.path.exists(xml_temp_dir):
        os.makedirs(xml_temp_dir)

    num_aug = 2000
    num_processes = min(mp.cpu_count(), 6)  # 使用CPU核心数，最多8个进程

    # 获取所有图像文件
    img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
    
    # 准备任务参数
    tasks = []
    for i in range(num_aug):
        img_name = random.choice(img_files)
        tasks.append((img_name, img_path, mask_path, mask_path_, save_img_path, xml_temp_dir, i))

    # 使用多进程处理
    successful_count = 0
    
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=num_aug, desc='正在生成') as pbar:
            for result in pool.imap_unordered(process_single_image, tasks):
                if result is not None:
                    successful_count += 1
                pbar.update(1)

    # 合并所有XML片段
    output_xml_file = os.path.join(save_xml_path, 'annotations_auged.xml')
    merge_xml_files(xml_temp_dir, output_xml_file)
    
    # 清理临时目录
    try:
        os.rmdir(xml_temp_dir)
    except:
        pass
    
    print(f"成功生成 {successful_count} 个增强样本")

if __name__ == '__main__':
    # 在Windows上使用多进程必须要有这个保护
    main()