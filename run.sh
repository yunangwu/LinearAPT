trial_count=10000

python experiment.py --sampler uniform \
                     --threshold 0 \
                     --precision 0.01 \
                     --trial "$trial_count" \
                     --budget 40 80 120 160 200

python experiment.py --sampler uniform_low_dim \
                     --threshold 0 \
                     --precision 0.01 \
                     --trial "$trial_count" \
                     --budget 40 80 120 160 200

python experiment.py --sampler iris \
                     --precision 0.1 \
                     --trial "$trial_count" \
                     --budget 200 250 300 350 400

python experiment.py --sampler wine \
                     --precision 0.1 \
                     --trial "$trial_count" \
                     --budget 200 250 300 350 400

python plot.py --file_path run/result_uniform.json
python plot.py --file_path run/result_uniform_low_dim.json
python plot.py --file_path run/result_iris.json
python plot.py --file_path run/result_wine.json