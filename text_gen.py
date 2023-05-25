import ray
from typing import Dict
import numpy as np
import time
import re
import pandas as pd
import numpy as np
from typing import Dict
from transformers import pipeline


# ===================== Tune params to scale and run faster ===================== 
MAX_WORKERS: int = 1
NUM_GPUS_PER_WORKER: float = 1
NUM_PARTITIONS = 1
BATCH_SIZE = 128


#=====================  Data source ===================== 
s3_uri = "s3://air-example-data/prompts.txt"


def load_ray_dataset():
    ds = ray.data.read_text(s3_uri)
    return ds


def remove_all_trailing_ellipses(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cleaned_text_batch = [re.sub("\.\.\.$", '', record) for record in batch["text"]]
    return {"text": np.array(cleaned_text_batch)}

class HuggingFacePredictor:
    def __init__(self):
        self.model = pipeline('text-generation', model='gpt2', device=0)

    def __call__(self, batch: Dict[str, np.ndarray]):        
        model_out = self.model(list(batch["text"]), 
                               max_length=30, 
                               num_return_sequences=1, 
                               pad_token_id=self.model.tokenizer.eos_token_id)
        batch["output"] = [sequence[0]["generated_text"] for sequence in model_out]
        return batch

def partition_data(ds):
    # https://docs.ray.io/en/master/data/batch_inference.html#getting-predictions-with-ray-data
    # https://docs.ray.io/en/master/data/batch_inference.html#batch-inference-config
    return ds.repartition(NUM_PARTITIONS)

def run_batch_inference(dataset, sample_fraction=100.0):
    dataset = partition_data(dataset)
    print ("====================== Partitioning done ==============")

    if sample_fraction < 100.0:
        dataset = dataset.random_sample(sample_fraction)


    start_time = time.time()
    scale = ray.data.ActorPoolStrategy(min_size=1, max_size=MAX_WORKERS)
    predictions = dataset.map_batches(HuggingFacePredictor, compute=scale, num_gpus=1, batch_size=BATCH_SIZE)
    print(predictions.take(5))
    print("Time taken: %s seconds" % (time.time() - start_time))

## Activity 1: Load the prompts data
ds = load_ray_dataset()
print("Sample input: ")
ds.show(2)

## Activity 2: Preprocess the prompts
ds = ds.map_batches(remove_all_trailing_ellipses)
print("Sample input after processing: ")
ds.show(2)

#Activity 3: Run batch inference
run_batch_inference(ds, sample_fraction=100.0)