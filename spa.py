with open("data/surface_map_file_freebase_complete_all_mention") as f:
    lines = f.readlines()

# save top-1000 lines
with open("data/surface_map_file_freebase_complete_all_mention_top10000", "w") as f:
    for line in lines[:10000]:
        f.write(line)
    