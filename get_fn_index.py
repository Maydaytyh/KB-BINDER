from simcse import SimCSE
import torch
import pickle 
import joblib

model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
model.device = torch.device("cuda:6")

with open("data/surface_map_file_freebase_complete_all_mention") as f:
    lines = f.readlines()
name_to_id_dict = {}
for line in lines:
    info = line.split("\t")
    name = info[0]
    score = float(info[1])
    mid = info[2].strip()
    if name in name_to_id_dict:
        name_to_id_dict[name][mid] = score
    else:
        name_to_id_dict[name] = {}
        name_to_id_dict[name][mid] = score
all_fns = list(name_to_id_dict.keys())
# tokenized_all_fns = [fn.split() for fn in all_fns]
print(len(all_fns))
model.build_index(all_fns,batch_size=64)
# joblib.dump(model.index, "data/fn_index_1.pkl")
# with open("data/fn_index_1.pkl", "wb") as f:
#     pickle.dump(model.index, f)
# 假设我们想要将 model.index 分为大小为 1000 的块
chunk_size = 10000
num_chunks = len(model.index) // chunk_size + 1

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(model.index))
    chunk = model.index[start:end]
    joblib.dump(chunk, f"data/fn_index_{i+1}.pkl")