# 将包含句子的json文件转化为txt格式以便srilm处理，后续生成语言模型以计算forward ppl和backward ppl
import json

data_dir = './Senticap/dataset_senticap.json'
dataset = json.load(open(data_dir, 'r'))
# out_dir = './tools//text_srilm_po.txt'          # positive
out_dir = './tools/text_srilm_ne.txt'          # negative

num = 0
with open(out_dir, 'w') as f:
    for item in dataset:
        if item["style"] == "negative":
            for sub_item in item["captions"]:
                num += 1
                sentence_list = sub_item["caption"]
                sentence = ''
                for word in sentence_list:
                    sentence += word
                    sentence += ' '
                f.writelines(sentence+'\n')
print("Total sen: "+str(num))

f.close()