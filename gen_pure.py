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
    


with open(sys.argv[1] + "pure.s", "w", encoding="utf-8") as f2:
    for i in lines:
        f2.write(i)
                

