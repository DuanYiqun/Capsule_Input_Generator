import pandas as pd 

dfnorm = pd.read_csv('capsule_outputs.txt', sep='t')

print(dfnorm)

with open('capsule_outputs.tsv', 'w') as write_tsv:
    write_tsv.write(dfnorm.to_csv(sep='t', index=False))