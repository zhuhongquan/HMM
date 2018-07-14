import re


# 对人工分词文件进行处理，提取出词和对应的词性，存放为二维列表
def data_processing(datafile):
    with open(datafile, "r", encoding="utf-8") as datafile:
        result_list = []
        for row in datafile.readlines():
            if row == "\n":
                continue
            else:
                word = re.findall(r'^\d+\s(.+?)\s_\s', row)
                tag = re.findall(r'_\s([A-Z]+?)\s_\s_\s\d+', row)
                temp_list = []
                temp_list.extend(word)
                temp_list.extend(tag)
                result_list.append(temp_list)
    return result_list


# 对用HMM完成的词性标注文件进行处理，将词性放到列表里
def data_processing_2(datafile):
    with open(datafile, "r", encoding="utf-8") as datafile:
        test_list = []
        for row in datafile.readlines():
            if row == "\n":
                continue
            else:
                tag = re.findall(r'^.+_(.+)', row)
                test_list.extend(tag)
    return test_list


def main():
    list1 = data_processing("dev.conll")  # 人工分词文件
    list2 = data_processing_2("out.txt")  # HMM分词文件
    count_right = 0
    for i in range(len(list1)):
        if list1[i][1] == list2[i]:
            count_right += 1  # 如果两个词性标注一致，则判断为正确
    print(count_right/len(list1))  # 输出正确率


main()
