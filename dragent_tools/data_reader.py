import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import math
import re

# 定义CSV文件路径（根据实际情况修改）
TYPHOON_DATA_PATH = '其它_DATA2021000024_1945-2024.csv'


def extract_english_intensity(intensity_str):
    """从强度描述中提取英文缩写"""
    if not isinstance(intensity_str, str):
        return 'UN'
    match = re.search(r'\(([A-Z]+)\)', intensity_str)
    return match.group(1) if match else 'UN'


def load_typhoon_data(file_path):
    """
    从CSV文件加载台风数据
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return None

        # 中文表头与英文列名映射
        cn2en = {
            '台风编号': 'typhoon_id',
            '台风中文名称': 'name_cn',
            '台风英文名称': 'name_en',
            '台风起始时间': 'start_time',
            '台风结束时间': 'end_time',
            '当前台风时间': 'current_time',
            '经度': 'lon',
            '纬度': 'lat',
            '台风强度': 'intensity_full',
            '台风等级': 'category',
            '风速': 'wind_speed',
            '气压': 'pressure',
            '移动方向': 'move_direction',
            '移动速度': 'move_speed',
        }
        # 只用utf-8编码，且强制为中文表头
        try:
            df = pd.read_csv(
                file_path,
                header=0,
                encoding='utf-8',
                parse_dates=['台风起始时间', '台风结束时间', '当前台风时间'],
                on_bad_lines='skip'
            )
            df = df.rename(columns=cn2en)
        except Exception as e:
            print(f"台风数据文件读取失败: {e}")
            return None

        # 数据清洗和转换
        # 0. 确保所有需要的列都存在
        required_cols = ['lat', 'lon', 'wind_speed', 'pressure', 'move_speed',
                         'name_cn', 'name_en', 'move_direction', 'intensity_full', 'typhoon_id', 'current_time',
                         'start_time', 'end_time']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        # 1. 处理经纬度
        for col in ['lat', 'lon']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. 处理数值列
        for col in ['wind_speed', 'pressure', 'move_speed']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. 从完整强度描述中提取英文缩写
        if 'intensity_full' in df.columns:
            df['intensity'] = df['intensity_full'].apply(extract_english_intensity)
        else:
            df['intensity'] = 'UN'

        # 4. 处理文本列
        for col in ['name_cn', 'name_en', 'move_direction']:
            if col in df.columns:
                df[col] = df[col].fillna('未知').astype(str)
            else:
                df[col] = '未知'

        # 5. 处理台风标识列
        if 'typhoon_id' in df.columns:
            df['typhoon_id'] = df['typhoon_id'].astype(str)
        else:
            df['typhoon_id'] = '未知'

        # 删除无效数据
        initial_count = len(df)
        df = df.dropna(subset=['lat', 'lon', 'current_time'])
        return df

    except Exception as e:
        print(f"[ERROR] 加载台风数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回空DataFrame而不是None，避免主流程误判
        return pd.DataFrame()


def haversine(lon1, lat1, lon2, lat2):
    """
    使用Haversine公式计算两点间的大圆距离（单位：公里）
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球半径，单位公里
    return c * r


def find_nearest_typhoon(lat, lon, target_time, typhoon_df, time_window_hours=6):
    """
    查找距离指定位置和时间最近的台风点
    """
    try:
        # 时间窗口：前后N小时
        time_window = timedelta(hours=time_window_hours)
        start_time = target_time - time_window
        end_time = target_time + time_window

        # 筛选时间窗口内的台风点
        time_filtered = typhoon_df[
            (typhoon_df['current_time'] >= start_time) &
            (typhoon_df['current_time'] <= end_time)
            ].copy()

        if time_filtered.empty:
            return None

        # 只按空间距离找最近点
        time_filtered['distance'] = time_filtered.apply(
            lambda row: haversine(lon, lat, row['lon'], row['lat']),
            axis=1
        )
        nearest_idx = time_filtered['distance'].idxmin()
        return typhoon_df.loc[nearest_idx]

    except Exception as e:
        # print(f"查找最近台风失败: {str(e)}")
        return None


def typhoon_api(input_data):
    """
    主API函数：根据经纬度和时间返回台风信息
    """
    # 解析输入参数
    try:
        args = json.loads(input_data)['arguments']
        lat = float(args['latitude'])
        lon = float(args['longitude'])
        time_str = args.get('time', datetime.utcnow().strftime('%Y-%m-%d %H:%M'))
        try:
            target_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
        except ValueError:
            target_time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
    except Exception:
        return _typhoon_unknown_result()

    typhoon_df = load_typhoon_data(TYPHOON_DATA_PATH)
    if typhoon_df is None:
        return _typhoon_unknown_result()

    typhoon_point = find_nearest_typhoon(lat, lon, target_time, typhoon_df)
    if typhoon_point is None:
        return _typhoon_unknown_result()

    typhoon_id = typhoon_point['typhoon_id']
    full_typhoon = typhoon_df[typhoon_df['typhoon_id'] == typhoon_id]
    if full_typhoon.empty:
        return _typhoon_unknown_result()

    start_time = full_typhoon['start_time'].iloc[0]
    end_time = full_typhoon['end_time'].iloc[0]

    def safe_str(val, default="未知"):
        return str(val) if pd.notna(val) and str(val).strip() != '' else default

    def safe_float(val, default="未知", ndigits=1):
        try:
            if pd.notna(val):
                return round(float(val), ndigits)
        except Exception:
            pass
        return default

    def safe_time(val, default="未知"):
        try:
            if pd.notna(val):
                return val.strftime('%Y-%m-%d %H:%M')
        except Exception:
            pass
        return default

    def extract_chinese_intensity(full_str):
        return re.sub(r'\(.*?\)', '', str(full_str)).strip() if isinstance(full_str, str) else "未知"

    def extract_english_intensity_from_full(full_str):
        if not isinstance(full_str, str):
            return "Unknown"
        m = re.search(r'\(([A-Z]+)\)', full_str)
        return m.group(1) if m else "Unknown"

    wind = typhoon_point.get('wind_speed')
    wind_str = f"{safe_float(wind)} m/s" if pd.notna(wind) else "未知"
    pressure = typhoon_point.get('pressure')
    pressure_str = f"{safe_float(pressure, ndigits=0)} hPa" if pd.notna(pressure) else "未知"
    move_speed = typhoon_point.get('move_speed')
    move_speed_str = f"{safe_float(move_speed)} km/h" if pd.notna(move_speed) else "未知"
    category_val = typhoon_point.get('category')
    typhoon_category = safe_str(category_val) if pd.notna(category_val) and str(category_val).strip() else safe_str(
        typhoon_point.get('intensity'))

    return {
        "台风编号": safe_str(typhoon_id),
        "台风中文名称": safe_str(typhoon_point.get('name_cn')),
        "台风英文名称": safe_str(typhoon_point.get('name_en'), "Unknown"),
        "台风起始时间": safe_time(start_time),
        "台风结束时间": safe_time(end_time),
        "当前台风时间": safe_time(typhoon_point.get('current_time')),
        "经度": safe_float(typhoon_point.get('lon')),
        "纬度": safe_float(typhoon_point.get('lat')),
        "台风强度": extract_chinese_intensity(typhoon_point.get('intensity_full')),
        "台风英文等级": extract_english_intensity_from_full(typhoon_point.get('intensity_full')),
        "台风等级": typhoon_category,
        "风速": wind_str,
        "气压": pressure_str,
        "移动方向": safe_str(typhoon_point.get('move_direction')),
        "移动速度": move_speed_str,
        "查询点距离": safe_float(haversine(lon, lat, typhoon_point.get('lon'), typhoon_point.get('lat')))
    }


def _typhoon_unknown_result():
    return {
        "台风编号": "未知",
        "台风中文名称": "未知",
        "台风英文名称": "Unknown",
        "台风起始时间": "未知",
        "台风结束时间": "未知",
        "当前台风时间": "未知",
        "经度": "未知",
        "纬度": "未知",
        "台风强度": "未知",
        "台风英文等级": "Unknown",
        "台风等级": "未知",
        "风速": "未知",
        "气压": "未知",
        "移动方向": "未知",
        "移动速度": "未知",
        "查询点距离": "未知"
    }


# 示例使用
if __name__ == "__main__":
    test_input1 = json.dumps({
        "name": "typhoon_api",
        "arguments": {
            "latitude": 21.0,
            "longitude": 157.0,
            # "time": "1945-05-01 12:00"
            "time": "2015-10-04 20:00"
        }
    })
    result1 = typhoon_api(test_input1)
    print(json.dumps(result1, indent=2, ensure_ascii=False))
