import time
from nester import print_lol_nester, print_lol2_nester, print_lol3_nester

# list：列表就是数组
# 列表没有越界，因为列表是动态的，如果访问一个不存在的元素，Python会给出一个'IndexError'作为响应

movies = ["你的名字", "言叶之庭", "秒速5厘米"]
print(movies[0])
print(len(movies))
# 从列表末尾删除数据
movies.pop()
print(movies)
# 从列表末尾增加数字
movies.extend(["秒速五厘米", "新海城"])
print(movies)
# 移除
movies.remove("新海城")
print(movies)
# 增加
movies.insert(0, "新海城")
print(movies)
# 排序
movies.sort()
print(movies)

# 元组:不能变的list
tup1 = ('physics', 'chemistry', 1997, 2000)
tup2 = (1, 2, 3, 4, 5)
tup3 = "a", "b", "c", "d"

# 时间
ticks = time.time()
print("1970.1.1 12:00am :", ticks)

# 迭代
for each_flick in movies:
    print(each_flick)

# 函数
Titanic = ["泰坦尼克号", ["1912年4月15日", "1912年4月10日", ["Jack", "Rose"]]]

for item in Titanic:
    if isinstance(item, list):
        for nested_item in item:
            if isinstance(nested_item, list):
                for deeper_item in nested_item:
                    print(deeper_item)
            else:
                print(nested_item)
    else:
        print(item)

# 使用函数的递归实现嵌套列表的拆解
# Python3默认的递归深度不能超过100


def print_lol(the_list):
    for lol_item in the_list:
        if isinstance(lol_item, list):
            print_lol(lol_item)
        else:
            print(lol_item)

print_lol(Titanic)

# 模块
# Python包搜素引擎(PyPI)为Internet上的第三方Python模块提供了一个集中的存储库

print_lol_nester(Titanic)

# 内置函数(BIF)
# range(4)生成一个0～4的迭代器
for num in range(4):
    print(num)

# 增加制表符
print_lol2_nester(Titanic)

print_lol3_nester(Titanic, True, 0)


