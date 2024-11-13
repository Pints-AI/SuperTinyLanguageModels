"""
The main generate code
"""

import hydra
import torch

from models.build_models import build_model
from models.generators import build_generator


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
    input_text = "So you are writing a math textbook. You love your subject enough to put in the hours, and you probably have some ideas on how the standard presentation can be improved. You care about good pedagogy and want to engage your students. You know that writing a text for undergraduates requires a different style from writing a research paper or a scholarly article for colleagues. But how to achieve that style? A good first step is to adjust your linguistic goal from"
    
    idx = model.embedding_model.tokenize_input(
        input_string=input_text,
        add_eot=False,
        truncate=False
    )
    # tokenize input text 
    idx = torch.tensor(idx).unsqueeze(0).to(torch.device(model.device))

    eot_token = model.embedding_model.byte_tokenizer.eot_token
    num_max_tokens = 20
    for _ in range(num_max_tokens):
        # logits, _ = model.inference(idx)
        # decoded_text = model.embedding_model.byte_tokenizer.decode(idx.view(-1).tolist())
        # input(decoded_text)

        logits, _ = model(idx)
        print(idx.size())
        print(logits.size())
        input()
        # check shape
        # input(logits.size()) # expected size [S, S_c, H_c]

        # get the actual logits
        new_byte_logits = logits[0] #[0, -1, :, :] # [S_c, H_c]

        # print(logits[0])
        # input()
        print(new_byte_logits)
        # decode all greedily
        tokens = torch.argmax(new_byte_logits, dim=-1) # [S_c, 1]

        # # check for eot token
        # print(tokens==eot_token)
        # eot_token_index = tokens.where(tokens==eot_token)#, 1.0, 0.0) # TODO might not work like this
        # print(eot_token_index)
        mask = tokens<256

        new_tokens = tokens[mask]
        print(idx)
        print(new_tokens)
        # print(new_tokens)


        # decode new tokens
        decoded_text = model.embedding_model.byte_tokenizer.decode(new_tokens.tolist())
        input(decoded_text)



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
        print(idx.size())

        # print full decoded input
        decoded_text = model.embedding_model.byte_tokenizer.decode(idx.tolist())
        input(decoded_text)

    # decode
    decoded_text = model.embedding_model.byte_tokenizer.decode(idx.tolist())

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
