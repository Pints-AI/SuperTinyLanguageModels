"""
The main generate code
"""

import hydra
import torch
import tiktoken

from models.build_models import build_model
from models.generators import build_generator

from trainers.prepare import EmbedderPreProcessor


@hydra.main(config_path="configs", config_name="generate")
def main(cfg):
    """run the main eval loop"""

    # set the checkpoint path to absolute path
    cfg["model_ckpt"] = hydra.utils.to_absolute_path(cfg["model_ckpt"])

    # load checkpoint from the path
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    model, _ = build_model(
        checkpoint_path=cfg["model_ckpt"],
        # model_cfg=cfg['model'],
        device=device,
        attention_type=cfg["attention_type"]
    )
                        
    # put model into eval mode
    model.eval()

    # force model to device
    model.to(torch.device(device))


    # test generation
    text = "Long after the village lights had faded and the moon cast its silver glow over the empty fields, a figure moved silently along the forgotten road. Shadows stretched across the path, and the distant hoot of an owl echoed through the night. At the edge of the forest, where the trees stood like silent sentinels, there was a doorway carved into the ancient stone, hidden from all but the most watchful eyes. It was said that beyond the doorway lay secrets waiting to be uncovered, mysteries that only the brave could"    
    idx = model.embedding_model.tokenize_input(
        input_string=text,
        add_eot=False,
        truncate=False
    )

    canonical_tokenizer = tiktoken.get_encoding('o200k_base')
    preprocessor = EmbedderPreProcessor(model.embedding_model, canonical_tokenizer=canonical_tokenizer)

    # tokenize input text 
    idx = torch.tensor(idx).unsqueeze(0).to(torch.device(model.device)) # 1, 340
    
    
    num_max_chunks = 300
    for _ in range(num_max_chunks):
        delimitations = preprocessor.process({"text": text})['labels']
        delimitations = torch.tensor(delimitations).unsqueeze(0).to(torch.device(model.device))

        logits, _ = model(idx, delimitations) # batch, num chunks, 259

        # get the actual logits
        new_byte_logits = logits[0] # [259]

        # print(logits[0])
        # input()
        # decode all greedily
        tokens = torch.argmax(new_byte_logits) # [1]
        # print(tokens)
        # # check for eot token
        # print(tokens==eot_token)
        # eot_token_index = tokens.where(tokens==eot_token)#, 1.0, 0.0) # TODO might not work like this
        # print(eot_token_index)
        
        # print(f"Input tokens: {idx}\n")
        # print(f"Generated token: {tokens}\n")

        # concatenate them to the idx tensor
        idx = torch.cat((idx, tokens.view(1, 1, *tokens.shape)), dim=1)

        text = model.embedding_model.byte_tokenizer.decode(idx.squeeze().tolist())
        # print(text)
        # input(f"\n\nNext input: {decoded_text}")
    print(text)
if __name__ == "__main__":
    main()
