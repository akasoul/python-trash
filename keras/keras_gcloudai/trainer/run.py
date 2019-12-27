import os
cmd1="python model.py --job-dir=C:/Users/Anton/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/tester/files/jobr/EURUSD/ \
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



cmd2="python model.py \
--job-dir=C:/Users/Anton/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/tester/files/jobr/EURUSD/ \
--mode='test' \
--data-size=5000 \
--eval-size=0.2 \
--epochs=50000 \
--overfit-epochs=50000 \
--reduction-epochs=100 \
--ls-reduction-koef=0.7 \
--ls=0.001 \
--l1=0.00000 \
--l2=0.00000 \
--drop-rate=0.1"


os.system(cmd2)