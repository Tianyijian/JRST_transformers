import csv
import re
import os
from collections import Counter

train_data = "./data/Train_Data.csv"
train_data_process = "./data/Train_Data_Process.csv"
train_data_process2 = "./data/Train_Data_Process2.csv"
train_data_process3 = "./data/Train_Data_Process3.csv"
train_data_mark = "./data/Train_Data_Mark.csv"
train_data_mark3 = "./data/Train_Data_Mark3.csv"
test_data = "./data/Test_Data.csv"
test_data_process = "./data/Test_Data_Process.csv"
test_data_process2 = "./data/Test_Data_Process2.csv"
test_data_mark = "./data/Test_Data_Mark.csv"
ensemble_file = "BASELINE2_ensemble.csv"
r2_ensemble_file = "./res/BASELINE0_bert_0-1-2-3-4.csv "
r2_ensemble_res = "./ensemble/BASELINE0_bert_0-1-2-3-4_ensemble.csv "
r2_ensemble_res_1 = "./ensemble/BASELINE0_bert_0-1-2-3-4_ensemble1.csv"

r2_train_data = "./data2/Round2_train.csv"
r2_train_data_process = "./data2/r2_Train_Data_Process.csv"
r2_train_data_process2 = "./data2/r2_Train_Data_Process2.csv"
r2_train_data_process3 = "./data2/r2_Train_Data_Process3.csv"
r2_train_data_mark = "./data2/r2_Train_Data_Mark.csv"
r2_train_data_mark3 = "./data2/r2_Train_Data_Mark3.csv"
r2_test_data = "./data2/round2_test.csv"
r2_test_data_process = "./data2/r2_Test_Data_Process.csv"
r2_test_data_process2 = "./data2/r2_Test_Data_Process2.csv"
r2_test_data_mark = "./data2/r2_Test_Data_Mark.csv"

all_train_data = "./data3/ALL_train.csv"
all_train_data_process3 = "./data3/ALL_train_process3.csv"
all_train_data_mark = "./data3/All_Train_Data_Mark.csv"
all_train_data_process = "./data3/ALL_train_process.csv"


def train_statistics():
    """对训练数据进行统计分析"""
    entity_num = 0
    entity_num_neg = 0
    sent = 0  # 句子数
    sent_neg = 0  # 负句子数
    sent_pos = 0  # 正句子数
    sent_no_entity = 0  # 无实体句子数
    sent_less_length = 0  # 小于一定长度的句子数
    max_length = 512
    sent_format_wrong = 0
    text_same = 0  # 标题包含在文本中(或相同)的句子数
    total_length = 0
    # with open(all_train_data, "r", encoding="UTF-8") as f:
    with open(all_train_data_process, "r", encoding="UTF-8") as f:
        # reader = csv.reader(f, delimiter=",", quotechar='"')
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            # text_a = "".join(line[1:-3])  # 文本中出现 ','
            # text_a = line[1] + line[2]  # 文本中出现 ','
            if line[2] == line[1] or (line[1] in line[2] and line[1] != ""):
                text_a = line[2]
                text_same += 1
                # print("{}-{}: text_same:{} ".format(i + 1, text_same, str(line)))
            else:
                text_a = "".join(line[1:3])
            if len(line) > 6:
                print("wrong format {}: {}".format(line[0], len(line)))
                sent_format_wrong += 1
            if len(text_a) < max_length:
                sent_less_length += 1
            if line[-3] == "":
                sent_no_entity += 1
                # print("no entity:  {}".format(line[0]))
            else:
                entity_num += len([e for e in line[-3].split(";") if e != ""])
            if line[-2] == "1":
                sent_neg += 1
            if line[-1] != "":
                entity_num_neg += len([e for e in line[-1].split(";") if e != ""])
            if line[-2] == "0":
                sent_pos += 1
            sent += 1
            total_length += len(text_a)
        print("------------train data-------------")
        print("sent_no_entity:{}, sent:{}, sent_pos:{}, sent_neg:{}, entity num:{}, entity_num_neg:{} ".format(
            sent_no_entity, sent, sent_pos, sent_neg, entity_num, entity_num_neg))
        print("aver_length:{}, sent < {}: {}, ratio:{}".format(float(total_length) / sent, max_length, sent_less_length,
                                                               sent_less_length / float(sent)))
        print("sent_wrong_format:{},  text_same:{}".format(sent_format_wrong, text_same))


def test_statistics():
    """对测试数据进行统计分析"""
    entity_num = 0
    sent_no_entity = 0
    sent = 0
    sent_less_length = 0  # 小于一定长度的句子数
    max_length = 512
    text_same = 0
    total_length = 0
    sent_format_wrong = 0
    # with open(r2_test_data, "r", encoding="UTF-8") as f:
    with open(r2_test_data_process, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            # text_a = "".join(line[1:-1])  # 文本中出现 ','
            if len(line) != 4:
                print("wrong format {}: {}".format(line[0], len(line)))
                sent_format_wrong += 1
            if line[2] == line[1] or (line[1] in line[2] and line[1] != ""):
                text_a = line[2]
                text_same += 1
                # print("{}-{}: text_same:{} ".format(i + 1, text_same, str(line)))
            else:
                text_a = "".join(line[1:3])
            sent += 1
            total_length += len(text_a)
            if len(text_a) < max_length:
                sent_less_length += 1
            if line[-1] == "":
                sent_no_entity += 1
                # print("no entity:  {}".format(line[0]))
            else:
                entity_num += len([e for e in line[-1].split(";") if e != ""])
        print("------------test data-------------")
        print("sent:{}, sent_no_entity:{}, entity_num:{}".format(sent, sent_no_entity, entity_num))
        print("aver_length:{}, sent < {}: {}, ratio:{}".format(float(total_length) / sent, max_length, sent_less_length,
                                                               sent_less_length / float(sent)))
        print("sent_wrong_format:{},  text_same:{}".format(sent_format_wrong, text_same))


def load_test():
    with open(r2_test_data, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            lines.append(line)
        return lines[1:]


def data_process():
    """对数据进行清洗 增加 title 与 text 相同的数量"""
    # data = [train_data, test_data, train_data_process2, test_data_process2]
    data = [all_train_data, r2_test_data, all_train_data_process, r2_test_data_process]
    for i in range(2):
        with open(data[i], "r", encoding="UTF-8") as f:
            lines = []
            for line in f.readlines():
                line = line.strip(' ')
                line = re.sub('[“”]', '', line)  # 1917
                line = re.sub('http[:/\w\.]*', '', line)  # 961
                # line = re.sub('\?\?+', '', line)  # 5564
                line = re.sub('##+', '#', line)  # 3755
                # line = re.sub('\[超话\]', '', line)  # 4514
                line = re.sub('\{IMG:\d+\}', '', line)  # 1704
                # line = re.sub('\s\s+', ' ', line)  # 9847
                line = re.sub('\s\s\s+', ' ', line)  # 9847
                # line = re.sub('(\?\s?){2,}', '', line)  # 212
                line = re.sub('全文： \?', '', line)  # 1371
                lines.append(line)
        with open(data[i + 2], "w", encoding="UTF-8") as f:
            f.writelines(lines)


def data_process2():
    """对训练数据中的实体进行融合清洗，测试集不用   已弃用
    注： 清洗方法需要改进： 先确保负面实体列表，再删除包含的或者被包含的"""
    # data_process()
    output = []
    # with open(train_data_process2, "r", encoding="UTF-8") as f:
    # with open(test_data, "r", encoding="UTF-8") as f:
    # with open(train_data, "r", encoding="UTF-8") as f:
    with open(r2_train_data, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        sent_remove_e_num = 0  # 记录删除实体的句子数
        e_remove_num = 0  # 记录删除的实体数量
        e_add_num = 0  # 删除重复实体后，需要从neg_entity中添加的实体
        for (i, line) in enumerate(reader):
            if i == 0:
                output.append(line)
                continue
            # if i > 10:
            #     break
            if line[3] == "":
                print("no entity:  {}".format(line[0]))
                output.append(line)
            else:
                entity = [e.strip() for e in line[3].split(";") if e != ""]
                if "宜贷网(沪)" in entity:
                    entity.remove("宜贷网(沪)")
                fund = False
                entity_copy = entity.copy()
                for j, term in enumerate(entity_copy):
                    for k, term2 in enumerate(entity_copy):
                        if j != k and term in term2 and not term == term2:
                            fund = True
                            entity.remove(term)
                            e_remove_num += 1
                            # print(
                            #     "Remove: {}-{}: {} {} in {}".format(i + 1, e_remove_num, str([line[x] for x in [0, 3, -1]]),
                            #                                         term,
                            #                                         term2))
                            break
                if fund:
                    sent_remove_e_num += 1
                if len(line) == 6 and line[-1] != "":  # 训练集 保证负面实体列表中的实体一定在总的实体列表中
                    neg_entity = [e.strip() for e in line[-1].split(";") if e != ""]
                    for ne in neg_entity:
                        if ne not in entity:
                            print(
                                "Add: {}-{}: {} {} add to {}".format(i + 1, e_add_num,
                                                                     str([line[x] for x in [0, 3, -1]]),
                                                                     ne,
                                                                     ";".join(entity)))
                            entity.append(ne)
                            e_add_num += 1
                line[3] = ";".join(entity)
                output.append(line)
    # with open(train_data_process2, "w", encoding="UTF-8", newline='') as f:
    with open(r2_train_data_process2, "w", encoding="UTF-8", newline='') as f:
        # with open(test_data_process2, "w", encoding="UTF-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output)
    print("\nsent_remove_e_num:{}, e_remove_num:{}, e_add_num:{}".format(sent_remove_e_num, e_remove_num, e_add_num))


def data_process3():
    """对训练数据中的实体进行融合清洗，测试集不用
    注： 清洗方法： 先确保负面实体列表，再删除包含的或者被包含的"""
    del_e_dict = {}
    # data_process()
    output = []
    # with open(train_data_process2, "r", encoding="UTF-8") as f:
    # with open(test_data, "r", encoding="UTF-8") as f:
    # with open(train_data, "r", encoding="UTF-8") as f:
    with open(all_train_data_process, "r", encoding="UTF-8") as f:
        # with open(r2_train_data, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        sent_remove_e_num = 0  # 记录删除实体的句子数
        e_remove_num = 0  # 记录删除的实体数量
        e_add_num = 0  # 删除重复实体后，需要从neg_entity中添加的实体
        for (i, line) in enumerate(reader):
            if i == 0:
                output.append(line)
                continue
            # if i > 10:
            #     break
            if line[3] == "":
                # print("no entity:  {}".format(line[0]))
                output.append(line)
            else:
                entity = [e.strip() for e in line[3].split(";") if e != ""]

                if len(line) == 6 and line[-1] != "":  # 训练集 先确保负面实体列表
                    neg_entity = [e.strip() for e in line[-1].split(";") if e != ""]
                    fund = False
                    for ne in neg_entity:
                        entity_copy = entity.copy()
                        for j, term in enumerate(entity_copy):
                            if term not in neg_entity and (term in ne or ne in term):
                                fund = True
                                entity.remove(term)
                                # 将该实体计入实体删除字典中
                                if term not in del_e_dict:
                                    del_e_dict[term] = {"total": 1, ne: 1}
                                else:
                                    del_e_dict[term]["total"] += 1
                                    if ne not in del_e_dict[term]:
                                        del_e_dict[term][ne] = 1
                                    else:
                                        del_e_dict[term][ne] += 1
                                e_remove_num += 1
                                # print(
                                #     "Remove: {}-{}: {} del {} for {}".format(i + 1, e_remove_num,
                                #                                              str([line[x] for x in [0, 3, -1]]),
                                #                                              term,
                                #                                              ne))
                    for ne in neg_entity:
                        if ne not in entity:
                            print(
                                "Add: {}-{}: {} {} add to {}".format(i + 1, e_add_num,
                                                                     str([line[x] for x in [0, 3, -1]]),
                                                                     ne,
                                                                     ";".join(entity)))
                            entity.append(ne)
                            e_add_num += 1
                    if fund:
                        sent_remove_e_num += 1
                line[3] = ";".join(entity)
                output.append(line)
    with open(all_train_data_process3, "w", encoding="UTF-8", newline='') as f:
        # with open(r2_train_data_process3, "w", encoding="UTF-8", newline='') as f:
        # with open(test_data_process2, "w", encoding="UTF-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output)
    print("\nsent_remove_e_num:{}, e_remove_num:{}, e_add_num:{}".format(sent_remove_e_num, e_remove_num, e_add_num))

    print("-----------------del_e_dict----------")
    del_e_dict_sort = sorted(del_e_dict.items(), key=lambda d: d[1]['total'], reverse=True)
    # for k, j in del_e_dict_sort:
    #     print("{}:{}".format(k, str(del_e_dict[k])))
    return del_e_dict


def ensemble():
    """将多个结果进行融合"""
    res = {}
    j = 0
    for root, dirs, files in os.walk("res"):
        for i, file in enumerate(files):
            if "bert" in file or "roberta" in file:
                print("ensemble: " + file)
                with open(root + "/" + file, 'r', encoding='utf-8') as f:
                    res[j] = [line for line in csv.reader(f)][1:]
                    j += 1
    diff = [0] * j
    rate = [[0, 0] for i in range(j)]
    min_cnt = 0
    with open("BASELINE0_bert_ensemble_0-1-2-3-4.csv", 'w', encoding="utf-8") as writer:
        writer.write("id,negative,key_entity\n")
        for i in range(len(res[0])):
            # for i in range(10):
            value = [int(res[k][i][1]) for k in range(j)]
            # print(value)
            # print(Counter(value).most_common(1))
            most_v = Counter(value).most_common(1)[0]
            diff[most_v[1] - 1] += 1

            if most_v[0] == 0:
                rate[most_v[1] - 1][0] += 1
                output_line = "{},{},{}\n".format(res[0][i][0], "0", "")
            elif most_v[0] == 1:
                rate[most_v[1] - 1][1] += 1
                entity = []
                for k in range(j):
                    entity.extend([term for term in res[k][i][2].split(";") if term != ""])
                # print(str([term for term in Counter(entity).most_common()]))
                neg_entity = []
                for term in Counter(entity).most_common():
                    if term[1] >= j // 2:
                        neg_entity.append(term[0])
                output_line = "{},{},{}\n".format(res[0][i][0], "1", ";".join(neg_entity))
                if most_v[1] == j // 2 + 1:
                    # print("{}:{} {} {}".format(res[k][i][0], value, entity, ";".join(neg_entity)))
                    min_cnt += 1
                if len(neg_entity) == 0:  # 标记为1不能没有实体
                    print("neg==0 {}:{} {} {}".format(res[k][i][0], value, entity, ";".join(neg_entity)))
            writer.write(output_line)
    print("Files same value: {}, 0/1: {}".format(str(diff), str(rate)))
    print(min_cnt)


def data_test():
    """对原始数据进行实体的统计"""
    data_file = r2_train_data
    # data_file = r2_test_data
    # data_file = train_data
    # data_file = test_data
    e_dict = {}  # 实体字典
    e_num = 0
    neg_e_dict = {}  # 负面实体字典
    pos_e_dict = {}  # 正面实体字典
    neg_num = 0
    line_entity_with_qm = 0
    with open(data_file, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            if '?' in line[3]:
                line_entity_with_qm += 1
                print("Line:{}-{} {}".format(i, line_entity_with_qm, [line[x] for x in [0, 3, 4, 5]]))
            if line[3] != "":
                entity = line[3].split(";")
                e_num += len(entity)
                for e in entity:
                    if e not in e_dict:
                        e_dict[e] = 1
                    else:
                        e_dict[e] += 1
                if len(line) == 6 and line[-1] != "":  # 有负面实体的训练集
                    neg_entity = line[-1].split(";")
                    neg_num += len(neg_entity)
                    for ne in neg_entity:
                        if ne not in neg_e_dict:
                            neg_e_dict[ne] = 1
                        else:
                            neg_e_dict[ne] += 1
                    for e in entity:  # 将非负面的加入正实体集
                        if e not in neg_entity:
                            if e not in pos_e_dict:
                                pos_e_dict[e] = 1
                            else:
                                pos_e_dict[e] += 1
                else:  # 无负面实体的数据，将实体集全部加入正实体集
                    for e in entity:
                        if e not in pos_e_dict:
                            pos_e_dict[e] = 1
                        else:
                            pos_e_dict[e] += 1
    print(
        "-----------entity: {}, pos/neg:{}/{}, freq:{}--------------".format(len(e_dict), len(pos_e_dict),
                                                                             len(neg_e_dict), e_num))
    print("line_entity_with_qm: {}".format(line_entity_with_qm))
    e_dict_sort = sorted(e_dict, key=e_dict.__getitem__, reverse=True)
    neg_e_dict_sort = sorted(neg_e_dict, key=neg_e_dict.__getitem__, reverse=True)
    pos_e_dict_sort = sorted(pos_e_dict, key=pos_e_dict.__getitem__, reverse=True)
    output = []
    for (i, k) in enumerate(e_dict_sort):
        if data_file == r2_train_data and i < len(pos_e_dict) and i < len(neg_e_dict):
            output_line = "{},{},{},{},{},{}\n".format(k, e_dict[k], pos_e_dict_sort[i], pos_e_dict[
                pos_e_dict_sort[i]], neg_e_dict_sort[i], neg_e_dict[neg_e_dict_sort[i]])
        elif data_file == r2_test_data:
            output_line = "{},{},{},{}\n".format(k, e_dict[k], pos_e_dict_sort[i], pos_e_dict[
                pos_e_dict_sort[i]])
        else:
            break
        output.append(output_line)
    with open("./data2/test_entity_statistic.csv", "w", encoding="utf_8_sig") as f:
        f.write("entity, ,pos_entity, , neg_entity, ,\n")
        f.writelines(output)


def ensemble_test():
    """对结果进行实体上的合并"""
    cnt = 0  # 记录移除实体的句子数
    remove_cnt = 0  # 记录总共移除的实体总数
    qs_cnt = 0  #
    out_put = []
    with open(r2_ensemble_file, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            if line[1] == "1" and line[2] == "":
                print("wrong: {}".format(str(line)))
            if line[1] == "0" and line[2] != "":
                print("wrong: {}".format(str(line)))

            if line[2] != "":
                entity = [e.strip() for e in line[2].split(";")]
                fund = False
                entity_copy = entity.copy()
                for j, term in enumerate(entity_copy):
                    for k, term2 in enumerate(entity_copy):
                        if j != k and term in term2 and not term == term2:
                            fund = True
                            entity.remove(term)
                            remove_cnt += 1
                            print("{}-{} :{} {} in {}".format(i + 1, remove_cnt, str(line), term, term2))
                            break
                if fund:
                    cnt += 1
                    # break

                for j, term in enumerate(entity):
                    if "?" in term:
                        # entity[j] = re.sub('\?\?+', '', term)
                        # print("{}-{}?: {}-{}".format(i + 1, qs_cnt, str(line), entity[j]))
                        qs_cnt += 1
                output_line = "{},{},{}\n".format(line[0], line[1], ";".join(entity))
            else:
                output_line = "{},{},{}\n".format(line[0], line[1], line[2])
            out_put.append(output_line)

        print("sent_remove_cnt:{}, remove_entity_cnt: {}, qs_cnt:{}".format(cnt, remove_cnt, qs_cnt))
    with open(r2_ensemble_res, 'w', encoding="utf-8") as writer:
        writer.write("id,negative,key_entity\n")
        writer.writelines(out_put)


def ensemble_process():
    """对结果进行实体上的合并"""
    del_e_dict = data_process3()
    cnt = 0  # 记录移除实体的句子数
    remove_cnt = 0  # 记录总共移除的实体总数
    qs_cnt = 0  #
    out_put = []
    with open(r2_ensemble_file, "r", encoding="UTF-8") as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            if line[1] == "1" and line[2] == "":
                print("wrong: {}".format(str(line)))
            if line[1] == "0" and line[2] != "":
                print("wrong: {}".format(str(line)))

            if line[2] != "":
                entity = [e.strip() for e in line[2].split(";")]
                fund = False
                del_e = []  # 记录需要删除的实体对及权值
                entity_copy = entity.copy()
                for j, term in enumerate(entity_copy):
                    if term in del_e_dict:
                        for k, term2 in enumerate(entity_copy):
                            if j != k and term2 in del_e_dict[term]:
                                del_e.append([term, term2, del_e_dict[term][term2]])
                # for a in del_e:
                #     print(a)
                # print("---------")
                if len(del_e) > 0:
                    del_e_sort = sorted(del_e, key=lambda x: x[2], reverse=True)
                    for a in del_e_sort:
                        for b in del_e_sort:
                            if a[0] == b[1] and a[1] == b[0]:
                                del_e_sort.remove(b)
                                print(
                                    "{}-{} :{} del {} for {}, freq:{}  ---no---".format(i + 1, remove_cnt, str(line),
                                                                                        b[0], b[1],
                                                                                        b[2]))
                    # for a in del_e_sort:
                    #     print(a)
                    for a in del_e_sort:
                        # print(a[0])
                        if a[0] in entity:
                            entity.remove(a[0])
                            fund = True
                            remove_cnt += 1
                            print(
                                "{}-{} :{} del {} for {}, freq:{}".format(i + 1, remove_cnt, str(line), a[0], a[1],
                                                                          a[2]))
                entity_copy = entity.copy()
                for j, term in enumerate(entity_copy):
                    for k, term2 in enumerate(entity_copy):
                        if j != k and term in term2 and not term == term2:
                            fund = True
                            entity.remove(term)
                            remove_cnt += 1
                            print("{}-{} :{} {} in {}".format(i + 1, remove_cnt, str(line), term, term2))
                            break
                if fund:
                    cnt += 1

                for j, term in enumerate(entity):
                    if "?" in term:
                        # entity[j] = re.sub('\?\?+', '', term)
                        # print("{}-{}?: {}-{}".format(i + 1, qs_cnt, str(line), entity[j]))
                        qs_cnt += 1
                output_line = "{},{},{}\n".format(line[0], line[1], ";".join(entity))
            else:
                output_line = "{},{},{}\n".format(line[0], line[1], line[2])
            out_put.append(output_line)

    print("sent_remove_cnt:{}, remove_entity_cnt: {}, qs_cnt:{}".format(cnt, remove_cnt, qs_cnt))
    with open(r2_ensemble_res_1, 'w', encoding="utf-8") as writer:
        writer.write("id,negative,key_entity\n")
        writer.writelines(out_put)


def data_mark():
    """将文本中的实体进行标记"""
    # data = [train_data_process2, test_data, train_data_mark, test_data_mark]
    # data = [r2_train_data_process2, r2_test_data, r2_train_data_mark, r2_test_data_mark]
    # data = [r2_train_data_process3, train_data_process3, r2_train_data_mark3, train_data_mark3]
    data = [all_train_data_process3, r2_test_data_process, all_train_data_mark, r2_test_data_mark]
    for j in range(2):
        with open(data[j], "r", encoding="UTF-8") as f:
            output = []
            write_line_num = 0
            reader = csv.reader(f)
            for (i, line) in enumerate(reader):
                if i == 0:
                    output.append(["id", "text", "entity", "label"])
                    continue
                # if i > 10:
                #     break;
                if line[2] == line[1] or (line[1] in line[2] and line[1] != ""):
                    text_a = line[2]
                else:
                    text_a = "".join(line[1:3])
                if line[3] == "":
                    print("no entity:  {}".format(line[0]))
                    output_line = [line[0], text_a, "", "0"]
                    output.append(output_line)
                    write_line_num += 1
                else:
                    entity = line[3].split(";")
                    if len(line) == 6 and line[-1] != "":
                        neg_entity = line[-1].split(";")
                    else:
                        neg_entity = []

                    for e in entity:
                        if e == "":
                            print("e is null: {}, {}".format(line[0], e))
                            continue
                        text = text_a
                        index = text.find(e)
                        while index != -1:
                            text = text[0:index] + "©" + text[index:index + len(e)] + "©" + text[index + len(e):]
                            index = text.find(e, index + len(e) + 2)
                            # print(index)
                            # print(text)
                        if e in neg_entity:
                            output_line = [line[0], text, e, "1"]
                        else:
                            output_line = [line[0], text, e, "0"]
                        # print(output_line)
                        output.append(output_line)
                        write_line_num += 1
        with open(data[j + 2], "w", encoding="UTF-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(output)
        print(write_line_num)


def prepare_data():
    """准备训练数据"""
    data_process()
    train_statistics()
    test_statistics()
    data_process3()
    data_mark()


if __name__ == '__main__':
    # data_process()
    # train_statistics()
    # test_statistics()
    # data_process3()
    # ensemble()
    # data_test()
    # ensemble_test()
    ensemble_process()
    # data_mark()
    # data_test()
    # data_process2()
    # prepare_data()
