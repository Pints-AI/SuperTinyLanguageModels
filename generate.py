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
        device=device,
        attention_type=cfg["attention_type"]
    )
                        
    # put model into eval mode
    model.eval()

    # force model to device
    model.to(torch.device(device))


    # test generation
    input_text = "Long after the village lights had faded and the moon cast its silver glow over the empty fields, a figure moved silently along the forgotten road. Shadows stretched across the path, and the distant hoot of an owl echoed through the night. At the edge of the forest, where the trees stood like silent sentinels, there was a doorway carved into the ancient stone, hidden from all but the most watchful eyes. It was said that beyond the doorway lay secrets waiting to be uncovered, mysteries that only the brave could"    
    idx = model.embedding_model.tokenize_input(
        input_string=input_text,
        add_eot=False,
        truncate=False
    )

    canonical_tokenizer = tiktoken.get_encoding('o200k_base')
    preprocessor = EmbedderPreProcessor(model.embedding_model, canonical_tokenizer=canonical_tokenizer)
    delimitations = preprocessor.process({"text": input_text})['labels']

    # tokenize input text 
    idx = torch.tensor(idx).unsqueeze(0).to(torch.device(model.device)) # 1, 340
    delimitations = torch.tensor(delimitations).unsqueeze(0).to(torch.device(model.device))

    original_idx_shape = idx.shape
    
    eot_token_id = model.embedding_model.eot_token
    num_max_chunks = 40
    for _ in range(num_max_chunks):
        # logits, _ = model.inference(idx)
        # decoded_text = model.embedding_model.byte_tokenizer.decode(idx.view(-1).tolist())
        # input(decoded_text)

        logits, _ = model(idx, delimitations) # batch, num chunks, chunk length, 259
        # check shape
        # input(logits.size()) # expected size [S, S_c, H_c]

        # get the actual logits
        new_byte_logits = logits[0][-1] #[0, -1, :, :] # [S_c, H_c]

        # print(logits[0])
        # input()
        # decode all greedily
        tokens = torch.argmax(new_byte_logits, dim=-1) # [S_c, 1]

        # # check for eot token
        # print(tokens==eot_token)
        # eot_token_index = tokens.where(tokens==eot_token)#, 1.0, 0.0) # TODO might not work like this
        # print(eot_token_index)


        first_eot_id_idx = (tokens == eot_token_id).nonzero(as_tuple=True)[0][0]
        new_tokens = tokens[:first_eot_id_idx]
        
        print(f"Input tokens: {idx}\n")
        print(f"Generated tokens: {tokens}\n")
        print(f"Masked tokens: {new_tokens}\n")
        # print(new_tokens.shape)


        # decode new tokens
        decoded_text = model.embedding_model.byte_tokenizer.decode(new_tokens.tolist())
        # input(f"decoded chunk: {decoded_text}")



        # print(tokens)
        # eot_token_index = ((tokens==eot_token).nonzero(as_tuple=True)[0])
        # print(eot_token_index)

        # truncate up to (not including) eot token 
        # new_tokens = tokens[:eot_token_index]
        # print(new_tokens)
        

        # decode for easier supervision
        # decoded = model.embedding_model.byte_tokenizer.decode(new_tokens.tolist())
        # input(decoded)

        # concatenate them to the idx tensor
        idx = torch.cat((idx, new_tokens.unsqueeze(0)), dim=1)
        new_delimitations = torch.zeros_like(new_tokens)
        new_delimitations[-1] = 1 # set 1 to the last char of the chunk
        delimitations = torch.cat((delimitations, new_delimitations.unsqueeze(0)), dim=1)
        # print full decoded input

        decoded_text = model.embedding_model.byte_tokenizer.decode(idx.squeeze().tolist())
        # input(f"\n\nNext input: {decoded_text}")

    # decode
    num_new_tokens = idx.shape[1] - original_idx_shape[1]
    all_generated_chunks = idx[:, -num_new_tokens:]
    decoded_text = model.embedding_model.byte_tokenizer.decode(all_generated_chunks.squeeze().tolist())

    # print the amazing new text
    print(decoded_text)





    # generator = build_generator(
    #     model=model,
    #     generate_cfg=cfg["generator"],
    #     device=device
    # )

    # # generate the text
    # for _ in range(5): # generate 5 samples
    #     generated_text = generator.default_generate(
    #         input_text=cfg["generator"]["input_text"]
    #     )
    #     print("".join(generated_text))

 
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
    # pylint: enable=no-value-for-parameter
