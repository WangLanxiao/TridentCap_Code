import torch
import os
import json
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_ppl(result_dir):
    with open(result_dir, 'rb') as f:
        while True:
            line = f.readline()
            line = line.decode('utf-8')
            if not line:
                break
            last_line = line
    tokens = last_line.split()
    ppl = float(tokens[-3])
    ppl1 = float(tokens[-1])
    print('ppl : ', ppl)
    print('ppl1 : ', ppl1)
    return ppl,ppl1


PPL_LM=[
        './tools/text_srilm_ro.lm',
        ./tools/text_srilm_fu.lm',
        ./tools/text_srilm_po.lm',
        './tools/text_srilm_ne.lm',
        ]

id=0
MODE=['ro','fu','pos','neg']
MODE=MODE[id]
name=['romantic','funny','positive','negative']
json_result='./results/output_'+name[id]+'.json'

res_data= json.load(open(json_result, 'r'))
txt_result=json_result.split('.json')[0]+'.txt'
ppl_path = json_result.split('.json')[0]+'_ppl'


if MODE=='ro':
    gt_txt=PPL_LM[0]
elif MODE=='fu':
    gt_txt=PPL_LM[1]
elif MODE=='pos':
    gt_txt=PPL_LM[2]
elif MODE=='neg':
    gt_txt=PPL_LM[3]

print("Building txt format ...")
with open(txt_result, 'w') as f:
    for item in tqdm(res_data):
        sentence_list = item['caption'].split(' ')
        sentence = ''
        for word in sentence_list:
            sentence += word
            sentence += ' '
        f.writelines(sentence+'\n')
f.close()

os.system('./tools/srilm-1.7.1/bin/i686-m64/ngram -ppl ' + txt_result + ' -order 3 -lm '+gt_txt+' > ' + ppl_path)
ppl,ppl1 = read_ppl(ppl_path)

