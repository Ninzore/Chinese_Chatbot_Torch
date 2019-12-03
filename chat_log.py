import csv
import re

raw_conv = 'D:/test1.txt'
prepreprocess = 'D:/prepreprocess.csv'
gen_prepre = False

def word_filter(raw):
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
                # 2018-02-01 0:12:00 ST AR-15(790667626)
                if info:
                    name = info.group('name')
                    date = info.group('date')
                    time = info.group('time')
                    uid = info.group('uid').strip("<").strip("(").strip(")").strip(">")
                    if uid == "771114514" or name == "系统消息":
                        continue
                    data = [date, time, name, uid]
                    continue
                else:
                    if re.search(r'给我康康', line) or re.search(r'让我康康', line) or re.search(r'^\[图片\]$', line) or re.search(r'^\[表情\]$', line):
                        continue
                    filttered = word_filter(line)
                    data.append(filttered.strip("\n"))
                    writer.writerow(data)
                    # print(data)
            else:
                pass