from simcse import SimCSE
import torch
import pickle 


model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
model.batch_size = 4096
model.device = torch.device("cuda:1")

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
model.build_index(all_fns,batch_size=128)

with open("data/fn_index.pkl", "wb") as f:
    pickle.dump(model.index, f)
