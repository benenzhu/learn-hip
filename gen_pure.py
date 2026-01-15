from re import L
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = f.readlines()


# remove comments
new_lines = []
for i in lines:
    if not i.strip().startswith(";"):
        new_lines.append(i)
lines = new_lines

# remove debug blocks: 
new_lines = []
start_with_Ldebug = False 
for i in lines:
    expect_list = [".Ldebug", ".Lloclists", "__hip_cuid", ".Lfunc_", ".Lrnglists_table", ".Lcu_begin", 
    "	.section	.rodata",
                   ".Linfo",
                   ".Laddr",


    ]
    new_start = False
    for j in expect_list:
        if i.startswith(j):
            new_start = True
            break
    if new_start:
        start_with_Ldebug = True
    elif i.startswith("."):
        start_with_Ldebug = False

    

    if not start_with_Ldebug:
        new_lines.append(i)
lines = new_lines 

print(len(lines))
lines = [i for i in lines if not i.startswith(".Ltmp")]
lines = [i for i in lines if len(i.strip())]

# if False:
if True:
    new_lines = []
    pre = None
    for i in lines:
        if i.strip().startswith(".loc"):
            pre = i.lstrip()
        else:
            if pre:
                i = i.rstrip() + "\t;" + pre
                pre = None
            new_lines.append(i)
        

    lines = new_lines
print(len(lines))


new_lines = []

cnt = 0
for i in lines:
    if i.strip().startswith("s_cbranch"):
        print("found s_cbranch", i.strip())
        new_lines.append(f".JUMP{i.strip().split()[1]}:\n")
    new_lines.append(i)
lines = new_lines
    
    
"""
v_mfma_f32_16x16x16_bf16 v[122:125], v[238:239], v[232:233], v[122:125] ; loc xxx 

->
v_mfma_f32_16x16x16_bf16 v122 v123 v124 v125, v238 v239, v232 v233, v122 v123 v124 v125; loc xxx 
"""

import re

# Step 1: Expand register ranges like v[122:125] -> v122 v123 v124 v125
def expand_reg_range(match):
    prefix = match.group(1)  # 'v' or 's'
    start = int(match.group(2))
    end = int(match.group(3))
    regs = [f"{prefix}{i}" for i in range(start, end + 1)]
    return " ".join(regs)

# Pattern to match v[start:end] only (not s[])
range_pattern = re.compile(r'(v)\[(\d+):(\d+)\]')

for i in range(len(lines)):
    lines[i] = range_pattern.sub(expand_reg_range, lines[i])

# Step 2: Zero-pad register numbers to 3 digits: v0 -> v000, v12 -> v012
def pad_reg_number(match):
    prefix = match.group(1)  # 'v' or 's'
    num = int(match.group(2))
    return f"{prefix}{num:03d}"

# Pattern to match standalone vgpr like v0, v12 (not sgpr)
# Use word boundary to avoid partial matches
reg_pattern = re.compile(r'\b(v)(\d+)\b')

for i in range(len(lines)):
    lines[i] = reg_pattern.sub(pad_reg_number, lines[i])

with open(sys.argv[1] + "pure.s", "w", encoding="utf-8") as f2:
    for i in lines:
        f2.write(i)
                

