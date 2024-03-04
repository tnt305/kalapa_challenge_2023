from transformers import AutoModel
import torch
import os

def convert_onnx(model, data_input, model_name='me5_small.onnx', logs_dir="./logs"):
    # Export the model
    os.makedirs(logs_dir, exist_ok=True)
    torch.onnx.export(model,               # model being run
                    (data_input["input_ids"], data_input["attention_mask"]), # model input (or a tuple for multiple inputs)
                    os.path.join(logs_dir, model_name),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=15,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input_ids', 'attention_mask'],   # the model's input names
                    output_names = ['logits'], # the model's output names
                    dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'sequence_len'},    # variable length axes
                                    'attention_mask' : {0 : 'batch_size', 1: 'sequence_len'},
                                    'logits' : {0 : 'batch_size'},
                                },
                    )
if __name__ == "__main__":
    data_input = {"input_ids": torch.rand((1, 512)).long(), "attention_mask": torch.rand((1, 512)).long()}
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
    model.to("cpu")
    model.eval()
    convert_onnx(model, data_input, logs_dir="./weight/me5_model")
    
