"""模块:嵌套列表的拆解输出."""


def print_lol_nester(the_list):
    # 使用函数的递归实现嵌套列表的拆解,
    # Python3默认的递归深度不能超过100.
    for lol_item in the_list:
        if isinstance(lol_item, list):
            print_lol_nester(lol_item)
        else:
            print(lol_item)


def print_lol2_nester(the_list, level=0):
    # 添加制表符，使输出结果更有层次
    # 为level增加缺省值
    for lol2_item in the_list:
        if isinstance(lol2_item, list):
            print_lol2_nester(lol2_item, level+1)
        else:
            for tab_stop in range(level):
                print("\t", end='')
            print(lol2_item)


def print_lol3_nester(the_list, indent=False, level=0):
    # 添加制表符，使输出结果更有层次
    # 为level增加缺省值
    # 增加是否使用制表符的开关
    for lol3_item in the_list:
        if isinstance(lol3_item, list):
            print_lol3_nester(lol3_item, indent, level+1)
        else:
            if indent:
                for tab_stop in range(level):
                    print("\t", end='')
            print(lol3_item)
