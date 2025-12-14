import argparse
from dfloat11mps.dfloat11 import DFloat11Model
from transformers import AutoTokenizer

def main(model_id, num_tokens, prompt):
    # Load model - will automatically use Swift Metal decoder and select MPS
    model = DFloat11Model.from_pretrained(model_id)

    # Run inference
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to('mps')
    
    # Model weights are decoded on-the-fly using Metal
    outputs = model.generate(**inputs, max_length=num_tokens, temperature=0.2)
    print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To run inference with DFLoat11.")
    parser.add_argument('--model_id', type=str, default='DFloat11/Qwen3-4B-DF11')
    parser.add_argument('--prompt', type=str, default='Question: What is a binary tree and its applications? Answer:')
    parser.add_argument('--num_tokens', type=int, default=512)
    args = parser.parse_args()

    main(args.model_id, args.num_tokens, args.prompt)