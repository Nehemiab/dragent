import requests
import geopandas as gpd
from datetime import datetime, timedelta
import os

def fetch_typhoon_multimodal_data(
    typhoon_id: str, 
    bbox: tuple, 
    time_range: tuple = None,
    data_types: list = None,
    cache_root: str = "./data_cache"
) -> dict:
    """
    获取并分类多模态台风应急数据
    
    参数:
        typhoon_id (str): 台风编号 (如: "2309-Doksuri")
        bbox (tuple): 地理边界框 (min_lon, min_lat, max_lon, max_lat)
        time_range (tuple): 时间范围 (start_time, end_time) 格式: "YYYY-MM-DD HH:MM"
        data_types (list): 指定需要的数据类型 (可选: ['space', 'sky', 'ground', 'human', 'network'])
        cache_root (str): 本地缓存根目录路径
        
    返回:
        dict: 分类后的多模态数据字典，结构:
        {
            "space": {  # 空基数据 (卫星/航空)
                "optical": GeoDataFrame/文件路径,  # 光学卫星影像
                "sar": GeoDataFrame/文件路径,      # 合成孔径雷达影像
                "infrared": GeoDataFrame/文件路径  # 红外卫星影像
            },
            "sky": {    # 天基数据 (气象/水文传感器)
                "weather": DataFrame,       # 气象站数据
                "hydrology": DataFrame      # 水文传感器数据
            },
            "ground": { # 地基数据 (地面观测)
                "uav_images": list,      # 无人机图像路径列表
                "field_reports": dict    # 现场报告数据
            },
            "human": {  # 人基数据 (社交媒体/人口密度)
                "social_media": DataFrame,    # 社交媒体数据
                "density_of_population": DataFrame   # 人口密度数据
            },
            "network": { # 网基数据 (基础设施)
                "power_grid": GeoDataFrame,    # 电网状态
                "traffic": GeoDataFrame,       # 交通状况
                "communication": GeoDataFrame  # 通信网络状态
            }
        }
    """
    # 创建缓存目录
    global cache_dir
    cache_dir = os.path.join(cache_root, typhoon_id)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"数据将保存到: {cache_dir}")
    
    # 默认获取全部数据类型
    if data_types is None:
        data_types = ['space', 'sky', 'ground', 'human', 'network']
    
    # 默认时间范围 (最近72小时)
    if time_range is None:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=72)
        time_range = (start_time.strftime("%Y-%m-%d %H:%M"), 
                      end_time.strftime("%Y-%m-%d %H:%M"))
        
    print(f"获取台风 {typhoon_id} 在区域 {bbox} 的数据")
    print(f"时间范围: {time_range[0]} 至 {time_range[1]}")
    
    # 初始化数据字典
    multimodal_data = {category: {} for category in data_types}
    
    # 1. 空基数据获取 (卫星/航空)
    if 'space' in data_types:
        try:
            print("获取空基数据...")
            multimodal_data['space']['optical'] = fetch_satellite_optical(typhoon_id, bbox, time_range)
            multimodal_data['space']['sar'] = fetch_satellite_sar(typhoon_id, bbox, time_range)
            multimodal_data['space']['infrared'] = fetch_satellite_infrared(typhoon_id, bbox, time_range)
        except Exception as e:
            print(f"空基数据获取失败: {str(e)}")
    
    # 2. 天基数据获取 (气象/水文传感器)
    if 'sky' in data_types:
        try:
            print("获取天基数据...")
            multimodal_data['sky']['weather'] = fetch_weather_data(bbox, time_range)
            multimodal_data['sky']['hydrology'] = fetch_hydrology_data(bbox, time_range)
        except Exception as e:
            print(f"天基数据获取失败: {str(e)}")
    
    # 3. 地基数据获取 (地面观测)
    if 'ground' in data_types:
        try:
            print("获取地基数据...")
            multimodal_data['ground']['uav_images'] = fetch_uav_images(bbox, time_range)
            multimodal_data['ground']['field_reports'] = fetch_field_reports(bbox, time_range)
        except Exception as e:
            print(f"地基数据获取失败: {str(e)}")
    
    # 4. 人基数据获取 (社交媒体/人口密度)
    if 'human' in data_types:
        try:
            print("获取人基数据...")
            multimodal_data['human']['social_media'] = fetch_social_media(typhoon_id, bbox, time_range)
            multimodal_data['human']['density_of_population'] = fetch_density_of_population(typhoon_id, bbox, time_range)
        except Exception as e:
            print(f"人基数据获取失败: {str(e)}")
    
    # 5. 网基数据获取 (基础设施)
    if 'network' in data_types:
        try:
            print("获取网基数据...")
            multimodal_data['network']['power_grid'] = fetch_power_grid_status(bbox)
            multimodal_data['network']['traffic'] = fetch_traffic_data(bbox, time_range)
            multimodal_data['network']['communication'] = fetch_communication_data(bbox, time_range)  
        except Exception as e:
            print(f"网基数据获取失败: {str(e)}")
    
    return multimodal_data

# 数据获取子函数
def fetch_satellite_optical(typhoon_id, bbox, time_range):
    """模拟获取光学卫星影像"""
    # 在实际项目中这里会下载真实图像
    # 这里创建模拟文件路径
    save_path = os.path.join(cache_dir, f"optical_{typhoon_id}.tif")
    
    # 创建模拟文件（实际使用中这里会是真实图像）
    with open(save_path, 'w') as f:
        f.write("模拟光学卫星图像数据")
    
    print(f"创建模拟光学图像: {save_path}")
    return save_path

def fetch_satellite_sar(typhoon_id, bbox, time_range):
    """模拟获取合成孔径雷达影像"""
    # 在实际项目中这里会下载真实图像
    # 这里创建模拟文件路径
    save_path = os.path.join(cache_dir, f"sar_{typhoon_id}.tif")
    
    # 创建模拟文件（实际使用中这里会是真实图像）
    with open(save_path, 'w') as f:
        f.write("模拟合成孔径雷达图像数据")
    
    print(f"创建模拟SAR图像: {save_path}")
    return save_path

def fetch_satellite_infrared(typhoon_id, bbox, time_range):
    """模拟获取红外卫星影像"""
    # 在实际项目中这里会下载真实图像
    # 这里创建模拟文件路径
    save_path = os.path.join(cache_dir, f"infrared_{typhoon_id}.tif")
    
    # 创建模拟文件（实际使用中这里会是真实图像）
    with open(save_path, 'w') as f:
        f.write("模拟红外卫星图像数据")
    
    print(f"创建模拟红外图像: {save_path}")
    return save_path

def fetch_weather_data(bbox, time_range):
    """
    获取气象站数据
    """
    # 模拟获取气象站数据
    weather_data = []  # 这里应该是实际的气象数据列表
    print(f"获取气象站数据，区域: {bbox}, 时间范围: {time_range}")
    return weather_data

def fetch_hydrology_data(bbox, time_range):
    """
    获取水文传感器数据
    """
    # 模拟获取水文传感器数据
    hydrology_data = []  # 这里应该是实际的水文数据列表
    print(f"获取水文传感器数据，区域: {bbox}, 时间范围: {time_range}")
    return hydrology_data

def fetch_uav_images(bbox, time_range):
    """
    获取无人机图像
    """
    # 模拟获取无人机图像
    uav_images = []  # 这里应该是实际的无人机图像路径列表
    print(f"获取无人机图像，区域: {bbox}, 时间范围: {time_range}")
    
    # 模拟创建一些图像文件
    for i in range(3):
        image_path = os.path.join(cache_dir, f"uav_image_{i}.jpg")
        with open(image_path, 'w') as f:
            f.write("模拟无人机图像数据")
        uav_images.append(image_path)
    
    return uav_images
    

def fetch_field_reports(bbox, time_range):
    """
    获取现场报告数据
    """
    # 模拟获取现场报告数据
    field_reports = {}  # 这里应该是实际的现场报告数据字典
    print(f"获取现场报告数据，区域: {bbox}, 时间范围: {time_range}")
    
    # 模拟创建一些报告
    field_reports['report_1'] = {
        'location': bbox,
        'time': time_range[0],
        'content': '模拟现场报告内容 1'
    }
    field_reports['report_2'] = {
        'location': bbox,
        'time': time_range[1],
        'content': '模拟现场报告内容 2'
    }
    
    return field_reports

def fetch_social_media(typhoon_id, bbox, time_range):
    """
    获取社交媒体数据
    """
    # 模拟获取社交媒体数据
    social_media_data = []  # 这里应该是实际的社交媒体数据列表
    print(f"获取社交媒体数据，台风ID: {typhoon_id}, 区域: {bbox}, 时间范围: {time_range}")
    
    # 模拟创建一些社交媒体帖子
    social_media_data.append({
        'id': 'post_1',
        'content': '模拟微博内容 1',
        'location': bbox,
        'time': time_range[0]
    })
    social_media_data.append({
        'id': 'post_2',
        'content': '模拟微博内容 2',
        'location': bbox,
        'time': time_range[1]
    })
    
    return social_media_data

def fetch_density_of_population(typhoon_id, bbox, time_range):
    """
    获取人口密度数据
    """
    # 模拟获取人口密度数据
    density_data = []  # 这里应该是实际的人口密度数据列表
    print(f"获取人口密度数据，台风ID: {typhoon_id}, 区域: {bbox}, 时间范围: {time_range}")

    # 模拟创建一些人口密度数据
    density_data.append({
        'id': 'density_1',
        'location': bbox,
        'time': time_range[0],
        'density': 1000  # 模拟人口密度值
    })

    return density_data

def fetch_traffic_data(bbox, time_range):
    """
    获取交通状况数据
    """
    # 模拟获取交通状况数据
    traffic_data = []  # 这里应该是实际的交通数据列表
    print(f"获取交通状况数据，区域: {bbox}, 时间范围: {time_range}")
    
    # 模拟创建一些交通数据文件
    for i in range(3):
        traffic_path = os.path.join(cache_dir, f"traffic_data_{i}.json")
        with open(traffic_path, 'w') as f:
            f.write(f"{{'id': 'traffic_{i}', 'status': 'normal', 'bbox': {bbox}}}")
        traffic_data.append(traffic_path)
    
    return traffic_data


def fetch_power_grid_status(bbox):
    """
    获取电网状态数据
    """
    # 模拟获取电网状态数据
    power_grid_data = gpd.GeoDataFrame()  # 这里应该是实际的电网状态数据
    print(f"获取电网状态数据，区域: {bbox}")
    
    # 模拟创建一些电网状态数据
    power_grid_data['id'] = ['grid_1', 'grid_2']
    power_grid_data['status'] = ['normal', 'outage']
    power_grid_data['geometry'] = [gpd.points_from_xy([bbox[0]], [bbox[1]]), gpd.points_from_xy([bbox[2]], [bbox[3]])]
    
    return power_grid_data

def fetch_communication_data(bbox, time_range):
    """
    获取通信网络状态数据
    """
    # 模拟获取通信网络状态数据
    communication_data = gpd.GeoDataFrame()  # 这里应该是实际的通信网络状态数据
    print(f"获取通信网络状态数据，区域: {bbox}, 时间范围: {time_range}")
    
    # 模拟创建一些通信网络状态数据
    communication_data['id'] = ['comm_1', 'comm_2']
    communication_data['status'] = ['operational', 'disrupted']
    communication_data['geometry'] = [gpd.points_from_xy([bbox[0]], [bbox[1]]), gpd.points_from_xy([bbox[2]], [bbox[3]])]
    
    return communication_data