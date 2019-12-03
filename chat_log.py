import csv
import re
#raw_conv是聊天记录的原始txt，需要先删除前面几行
#prepreprocess是稍微处理后格式化的聊天记录
raw_conv = 'D:/test1.txt'
prepreprocess = 'D:/prepreprocess.csv'
gen_prepre = False

def word_filter(raw):
    #在这里过滤文本
    raw = re.compile(r"ur[^a-zA-Z0-9\u4e00-\u9fa5\U0001F300-\U0001F64F]", re.UNICODE).sub("", raw)
    raw = re.compile(r"^\[图片\]").sub("", raw)
    raw = re.compile(r"\[图片\]").sub("pic", raw)
    return raw

with open(raw_conv,'r', encoding='utf-8') as raw_conv:
    with open(prepreprocess,'w', encoding='utf-8', newline="") as prepreprocess:
        headers = ["date", "time", "name", "uid", "sentence"]
        date = []
        writer = csv.writer(prepreprocess, delimiter='\t')
        writer.writerow(headers)
        for line in  raw_conv.readlines():
            # print(line_num)
            # print(len(line))
            # print(line)
            if len(line) > 1:
                info = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})\s(?P<time>\d{1,2}:\d{2}:\d{2})\s(?P<name>.*)(?P<uid>\(\d+\)|<.+>)', line)
                if info:
                    name = info.group('name')
                    date = info.group('date')
                    time = info.group('time')
                    uid = info.group('uid').strip("<").strip("(").strip(")").strip(">")
                    if uid == "123456" or name == "系统消息":
                        #去除需要无视的QQ号或者昵称
                        continue
                    data = [date, time, name, uid]
                    continue
                else:
                    if re.search(r'^\[图片\]$', line):
                        #如果是只有一幅图那就直接跳过这条消息
                        continue
                    filttered = word_filter(line)
                    data.append(filttered.strip("\n"))
                    writer.writerow(data)
                    # print(data)
            else:
                pass
