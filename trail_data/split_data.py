f = open('task1.jsonl', 'r')
content = f.readlines()
f.close()

f = open('train.jsonl','w')
f.writelines(content[:900])
f.close()

f = open('dev.jsonl','w')
f.writelines(content[900:950])
f.close()

f = open('test.jsonl','w')
f.writelines(content[950:])
f.close()