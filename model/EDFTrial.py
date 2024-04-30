from pyedflib import highlevel
import numpy as np
import scipy.io.wavfile
import xml.etree.ElementTree as ET
from random import uniform
from pathlib import Path

in_path = Path.cwd()
out_train = Path('/run/media/vaaru/T7/apnea/new_apnea_dataset/train')
out_val = Path('/run/media/vaaru/T7/apnea/new_apnea_dataset/val')

sample_length = 45
sample_rate = 48000
types = {'MixedApnea', 'ObstructiveApnea', 'NoApnea'}
type_counts = {event_type: 0 for event_type in types}

counter = 0
for rml_file in in_path.rglob("*.rml"):
    if(counter % 5):
        out_path = out_train
    else:
        out_path = out_val
        counter = 0
    counter += 1
    tree = ET.parse(rml_file)
    events = tree.getroot()[5][1]

    start_samples = []
    for event in events:
        if event.attrib['Type'] in types:
            event_type, start, duration = (event.attrib['Type'], event.attrib['Start'], event.attrib['Duration'])
            start_samples.append((f'{rml_file.stem}_{event_type}_{type_counts[event_type]}.wav', int((float(start) + uniform(0, float(duration) - sample_length)) * sample_rate)))
            type_counts[event_type] += 1

    base_name = rml_file.stem
    n_no_apnea = int(len(start_samples) / 4)

    for edf_file in in_path.rglob("*.edf"):
        # print(edf_file)
        if base_name not in edf_file.stem:
            continue
        print(edf_file)

        try:
            signals, signal_headers, header = highlevel.read_edf(str(edf_file), ch_names=['ECG I', 'Tracheal', 'Mic'])


            hour_offset = int(edf_file.stem[len(base_name)+1:len(base_name) + 4]) - 1

            mic = signals[2]

            for fname, start_sample in start_samples:
                if(start_sample / (sample_rate * 3600.0) > hour_offset and start_sample / (sample_rate * 3600.0) < hour_offset + 1):

                    adjusted = start_sample - (hour_offset * 3600 * sample_rate)
                    sample = mic[adjusted - 24000 * sample_length:adjusted + 24000 * sample_length];
                    if sample.shape[0] != sample_rate * sample_length:
                        print("main")
                        print(sample.shape[0], sample_rate * sample_length)
                        continue
                    scipy.io.wavfile.write(
                        out_path / fname,
                        sample_rate,
                        sample
                    )

            for i in range(n_no_apnea):
                count = 0
                while(True):
                    start_second = int(uniform(1, int(mic.shape[0] / 48000) - 100))
                    too_close = False
                    for _, apnea_start in start_samples:
                        if abs((start_second + hour_offset * 3600) * sample_rate - apnea_start) < 60 * sample_rate:
                            too_close = True
                            count += 1
                    if not too_close or count > 10:
                        break
                if count > 10:
                    continue           
            
                sample = mic[start_second*sample_rate:(start_second+sample_length)*sample_rate]
                if sample.shape[0] != sample_rate * sample_length:
                    print("no")
                    print(sample.shape[0], sample_rate * sample_length)
                    continue

                scipy.io.wavfile.write(
                    out_path / f'NoApnea_{type_counts["NoApnea"]}.wav',
                    sample_rate,
                    sample
                )
                type_counts['NoApnea'] += 1
        except Exception as e:
            print("ermmm")
            print(e)

