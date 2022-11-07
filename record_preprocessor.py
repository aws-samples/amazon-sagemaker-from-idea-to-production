import json
import random
import ast
import sys
import subprocess
import os

def preprocess_handler(inference_record):

    input_enc_type = inference_record.endpoint_input.encoding
    input_data = inference_record.endpoint_input.data
    output_data = inference_record.endpoint_output.data.rstrip("\n")
    eventmedatadata = inference_record.event_metadata
    custom_attribute = json.loads(eventmedatadata.custom_attribute[0]) if eventmedatadata.custom_attribute is not None else None

    if input_enc_type == "CSV":
        # don't include output data in the record for data quality monitor
        outputs = input_data
        return { str(f"_c{i}") : d for i, d in enumerate(outputs.split(",")) }
    else:
        raise ValueError(f"encoding type {input_enc_type} is not supported") 
        
        

