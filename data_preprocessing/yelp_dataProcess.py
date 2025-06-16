"""
Yelp数据预处理脚本
用于处理推荐系统的用户-商家交互数据，包括数据映射、过滤、分割和格式转换
"""

import datetime
import numpy as np
import json
import pickle
import logging
from scipy.sparse import csr_matrix
import time
import sys
import os

# ==================== 全局变量 ====================
# 用于记录数据中的最小和最大年份
min_year = 2022
max_year = 0

# ==================== 时间处理函数 ====================

def is_valid_year(year, month):
	"""
	检查年份是否在有效范围内
	保留2012-2022年的数据
	
	Args:
		year (int): 年份
		month (int): 月份
	
	Returns:
		bool: 是否为有效年份
	"""
	return year >= 2012 and year <= 2022


def transform_timestamp(timestamp_str):
	"""
	转换时间戳字符串并记录年份范围
	
	Args:
		timestamp_str (str): 时间戳字符串，格式为 'YYYY-MM-DD HH:MM:SS'
	
	Returns:
		float or None: 转换后的Unix时间戳，如果年份无效则返回None
	"""
	global min_year, max_year
	
	try:
		# 解析时间字符串
		time_obj = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
		year = time_obj.year
		month = time_obj.month
		
		# 更新年份范围
		min_year = min(min_year, year)
		max_year = max(max_year, year)
		
		# 检查年份是否有效
		if is_valid_year(year, month):
			return time_obj.replace(tzinfo=datetime.timezone.utc).timestamp()
		return None
	except ValueError:
		return None


# ==================== 数据映射函数 ====================

def map_ids_and_load_data(input_file):
	"""
	从CSV文件中读取数据，将用户ID和商家ID映射为数字索引
	
	Args:
		input_file (str): 输入CSV文件路径
	
	Returns:
		tuple: (交互数据列表, 用户数量, 商家数量)
	"""
	user_id_map = {}  # 用户ID到数字索引的映射
	business_id_map = {}  # 商家ID到数字索引的映射
	user_count = 0
	business_count = 0
	interactions = []  # 存储每个用户的交互数据
	
	with open(input_file, 'r') as file:
		for line in file:
			# 解析CSV行：用户ID,商家ID,...,时间戳
			fields = line.strip().split(',')
			user_id = fields[0]
			business_id = fields[1]
			timestamp = transform_timestamp(fields[-1])
			
			# 跳过无效时间戳
			if timestamp is None:
				continue
			
			# 为新用户分配索引
			if user_id not in user_id_map:
				user_id_map[user_id] = user_count
				interactions.append({})  # 为新用户创建交互字典
				user_count += 1
			
			# 为新商家分配索引
			if business_id not in business_id_map:
				business_id_map[business_id] = business_count
				business_count += 1
			
			# 记录交互
			user_idx = user_id_map[user_id]
			business_idx = business_id_map[business_id]
			interactions[user_idx][business_idx] = timestamp
	
	print(f'数据年份范围: {min_year} - {max_year}')
	return interactions, user_count, business_count


# ==================== 过滤条件函数 ====================

def filter_condition_1(count):
	"""第一轮过滤：交互次数 >= 15"""
	return count >= 15


def filter_condition_2(count):
	"""第二轮过滤：交互次数 >= 15"""
	return count >= 15


def filter_condition_3(count):
	"""第三轮过滤：交互次数 >= 15"""
	return count >= 15


# ==================== 数据过滤函数 ====================

def filter_sparse_data(interactions, user_count, business_count, 
					  user_filter_func, business_filter_func, filter_businesses=True):
	"""
	过滤稀疏数据，去除交互次数过少的用户和商家
	
	Args:
		interactions (list): 用户交互数据
		user_count (int): 用户数量
		business_count (int): 商家数量
		user_filter_func (function): 用户过滤条件函数
		business_filter_func (function): 商家过滤条件函数
		filter_businesses (bool): 是否过滤商家
	
	Returns:
		tuple: (过滤后的交互数据, 新用户数量, 新商家数量)
	"""
	# 第一步：确定要保留的用户和商家
	users_to_keep = set()
	businesses_to_keep = set()
	business_interaction_counts = [0] * business_count
	
	# 统计每个用户和商家的交互次数
	for user_idx in range(user_count):
		user_data = interactions[user_idx]
		user_interaction_count = 0
		
		for business_idx in user_data:
			business_interaction_counts[business_idx] += 1
			user_interaction_count += 1
		
		# 检查用户是否满足过滤条件
		if user_filter_func(user_interaction_count):
			users_to_keep.add(user_idx)
	
	# 检查商家是否满足过滤条件
	for business_idx in range(business_count):
		if not filter_businesses or business_filter_func(business_interaction_counts[business_idx]):
			businesses_to_keep.add(business_idx)
	
	# 第二步：重新构建过滤后的数据
	filtered_interactions = []
	new_user_count = 0
	new_business_count = 0
	new_business_id_map = {}
	
	for old_user_idx in range(user_count):
		if old_user_idx not in users_to_keep:
			continue
		
		# 为保留的用户分配新索引
		new_user_idx = new_user_count
		new_user_count += 1
		filtered_interactions.append({})
		
		user_data = interactions[old_user_idx]
		for old_business_idx in user_data:
			if old_business_idx not in businesses_to_keep:
				continue
			
			# 为保留的商家分配新索引
			if old_business_idx not in new_business_id_map:
				new_business_id_map[old_business_idx] = new_business_count
				new_business_count += 1
			
			new_business_idx = new_business_id_map[old_business_idx]
			filtered_interactions[new_user_idx][new_business_idx] = user_data[old_business_idx]
	
	return filtered_interactions, new_user_count, new_business_count


# ==================== 数据分割函数 ====================

def split_train_test(interactions, user_count, business_count):
	"""
	将数据分割为训练集和测试集
	随机选择10000个用户，将他们最新的交互作为测试数据
	
	Args:
		interactions (list): 用户交互数据
		user_count (int): 用户数量
		business_count (int): 商家数量
	
	Returns:
		tuple: (训练集交互数据, 测试集交互数据)
	"""
	test_user_count = 10000
	
	# 随机选择用户用于测试
	user_permutation = np.random.permutation(user_count)
	selected_test_users = user_permutation[:test_user_count]
	
	test_interactions = [None] * user_count
	exception_count = 0
	
	for user_idx in selected_test_users:
		user_businesses = []
		user_data = interactions[user_idx]
		
		# 收集用户的所有交互，按时间排序
		for business_idx in user_data:
			user_businesses.append((business_idx, user_data[business_idx]))
		
		if len(user_businesses) == 0:
			exception_count += 1
			continue
		
		# 按时间戳排序，选择最新的交互作为测试数据
		user_businesses.sort(key=lambda x: x[1])
		latest_business = user_businesses[-1][0]
		
		# 将最新交互移到测试集
		test_interactions[user_idx] = latest_business
		interactions[user_idx][latest_business] = None  # 从训练集中移除
	
	print(f'异常用户数: {exception_count}, 有效测试用户数: {np.sum(np.array(test_interactions) != None)}')
	return interactions, test_interactions


# ==================== 数据转换函数 ====================

def convert_to_sparse_matrix(interactions, user_count, business_count):
	"""
	将交互数据转换为稀疏矩阵格式
	
	Args:
		interactions (list): 用户交互数据
		user_count (int): 用户数量
		business_count (int): 商家数量
	
	Returns:
		csr_matrix: 稀疏交互矩阵
	"""
	rows, cols, data = [], [], []
	
	for user_idx in range(user_count):
		if interactions[user_idx] is None:
			continue
		
		user_data = interactions[user_idx]
		for business_idx in user_data:
			if user_data[business_idx] is not None:
				rows.append(user_idx)
				cols.append(business_idx)
				data.append(user_data[business_idx])
	
	sparse_matrix = csr_matrix((data, (rows, cols)), shape=(user_count, business_count))
	return sparse_matrix


# ==================== 主程序 ====================

def main():
	"""主程序：执行完整的数据预处理流程"""
	
	# 配置日志
	data_prefix = 'yelp/'
	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
					   format=log_format, datefmt='%m/%d %I:%M:%S %p')
	
	file_handler = logging.FileHandler(os.path.join('./yelp', 'log.txt'))
	file_handler.setFormatter(logging.Formatter(log_format))
	logger = logging.getLogger()
	logger.addHandler(file_handler)
	
	logger.info('开始Yelp数据预处理')
	
	# 步骤1：数据映射和加载
	interactions, user_count, business_count = map_ids_and_load_data(data_prefix + 'yelp.csv')
	logger.info(f'ID映射完成，用户数: {user_count}, 商家数: {business_count}')
	
	# 步骤2：多轮数据过滤
	filter_functions = [filter_condition_1, filter_condition_2, filter_condition_3]
	
	for round_idx in range(3):
		filter_businesses = True  # 每轮都过滤商家
		interactions, user_count, business_count = filter_sparse_data(
			interactions, user_count, business_count, 
			filter_functions[round_idx], filter_functions[round_idx], filter_businesses
		)
		print(f'第{round_idx + 1}轮过滤后: 用户数 {user_count}, 商家数 {business_count}')
	
	logger.info(f'稀疏数据过滤完成，用户数: {user_count}, 商家数: {business_count}')
	
	# 步骤3：数据分割
	train_interactions, test_interactions = split_train_test(interactions, user_count, business_count)
	logger.info('数据集分割完成')
	
	# 步骤4：转换为稀疏矩阵
	train_matrix = convert_to_sparse_matrix(train_interactions, user_count, business_count)
	print(f'训练矩阵非零元素数量: {len(train_matrix.data)}')
	logger.info('训练矩阵构建完成')
	
	# 步骤5：保存数据
	with open(data_prefix + 'trn_mat', 'wb') as file:
		pickle.dump(train_matrix, file)
	
	with open(data_prefix + 'tst_int', 'wb') as file:
		pickle.dump(test_interactions, file)
	
	logger.info('交互数据保存完成')


if __name__ == "__main__":
	main()