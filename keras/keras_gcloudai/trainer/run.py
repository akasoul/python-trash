import os
cmd="python trainModel.py --job-dir=C:/Users/antonvoloshuk/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/tester/files/jobr/EURUSD/ \
--mode='test' \
--data-size=5000 \
--eval-size=0.2 \
--epochs=30000 \
--overfit-epochs=5000\
--reduction-epochs=50000\
--ls-reduction-koef=0.95\
--ls=0.00005\
--l1=0.00001\
--l2=0.00001\
--drop-rate=0.15'"

os.system(cmd)